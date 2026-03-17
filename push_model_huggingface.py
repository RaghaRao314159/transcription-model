import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, WhisperConfig, WhisperFeatureExtractor

from audio_transcription_config import AudioTranscriptionConfig
from train_audio import AudioTranscriptionModel as TrainingAudioTranscriptionModel


DEFAULT_REPO_ID = "RaghaRao314159/transcription-models"
DEFAULT_STAGES = ("stage_a", "stage_b")
LEGACY_STAGE_FOLDERS = ("audio_stage_b",)
LEGACY_ROOT_FILES = (
    "audio_projector.bin",
    "chat_template.jinja",
    "config.json",
    "export_manifest.json",
    "loss_curves.png",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package Stage A and Stage B audio transcription checkpoints and push them to the Hugging Face Hub.",
    )
    parser.add_argument("--stage-a-dir", default="checkpoints/audio_stage_a")
    parser.add_argument("--stage-b-dir", default="checkpoints/audio_stage_b")
    parser.add_argument("--output-dir", default="artifacts/hf_audio_models")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-private", action="store_true")
    parser.add_argument("--commit-message", default="Upload stage_a and stage_b audio transcription packages")
    parser.add_argument("--whisper-model", default="openai/whisper-base")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--pool-kernel", type=int, default=4)
    parser.add_argument("--prompt-text", default="Transcribe:")
    parser.add_argument("--stages", default="stage_a,stage_b")
    parser.add_argument("--package-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--cleanup-legacy-layout", action="store_true")
    return parser.parse_args()


def parse_stages(stages_arg: str) -> list[str]:
    stages = []
    for raw in stages_arg.split(","):
        stage = raw.strip()
        if not stage:
            continue
        if stage not in DEFAULT_STAGES:
            raise ValueError(f"Unsupported stage: {stage!r}. Expected one of {DEFAULT_STAGES}.")
        if stage not in stages:
            stages.append(stage)
    if not stages:
        raise ValueError("No stages selected.")
    return stages


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def copy_stage_metadata(source_dir: Path, output_dir: Path):
    for name in ("chat_template.jinja",):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def maybe_write_chat_template(output_dir: Path, tokenizer):
    path = output_dir / "chat_template.jinja"
    if path.exists():
        return
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        path.write_text(chat_template)


def save_feature_extractor_and_generation_config(output_dir: Path, whisper_model: str, generation_source: str | Path):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
    feature_extractor.save_pretrained(output_dir)

    try:
        generation_config = GenerationConfig.from_pretrained(generation_source)
    except Exception:
        generation_config = None
    if generation_config is not None:
        generation_config.save_pretrained(output_dir)


def save_model_code(output_dir: Path):
    repo_root = Path(__file__).resolve().parent
    for filename in ("audio_transcription_config.py", "audio_transcription_model.py"):
        shutil.copy2(repo_root / filename, output_dir / filename)


def save_custom_config(output_dir: Path, args, tokenizer, stage_name: str):
    prompt_ids = tokenizer(args.prompt_text, add_special_tokens=True).input_ids

    config = AudioTranscriptionConfig(
        whisper_model_name=args.whisper_model,
        llm_model_name=args.llm_model,
        whisper_config=WhisperConfig.from_pretrained(args.whisper_model).to_dict(),
        llm_config=AutoConfig.from_pretrained(args.llm_model, trust_remote_code=True).to_dict(),
        pool_kernel=args.pool_kernel,
        prompt_text=args.prompt_text,
        prompt_ids=prompt_ids,
        architectures=["AudioTranscriptionModel"],
        training_stage=stage_name,
    )
    config.auto_map = {
        "AutoConfig": "audio_transcription_config.AudioTranscriptionConfig",
        "AutoModel": "audio_transcription_model.AudioTranscriptionModel",
    }
    config.save_pretrained(output_dir)


def validate_stage_package(output_dir: Path):
    required = [
        "audio_projector.bin",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Packaged model is missing required files in {output_dir}: {missing}")


def extract_projector_weights(stage_b_dir: Path, output_dir: Path):
    state_dict = load_file(stage_b_dir / "model.safetensors")
    projector_state = {}
    for key, value in state_dict.items():
        if key.startswith("audio_projector."):
            projector_state[key.replace("audio_projector.", "", 1)] = value
    if not projector_state:
        raise RuntimeError("No audio_projector weights found in Stage B checkpoint.")
    torch.save(projector_state, output_dir / "audio_projector.bin")


def build_stage_b_bundle(args, repo_layout_dir: Path):
    source_dir = Path(args.stage_b_dir)
    output_dir = repo_layout_dir / "stage_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("model.safetensors",):
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing Stage B file: {src}")
        shutil.copy2(src, output_dir / name)

    copy_stage_metadata(source_dir, output_dir)
    extract_projector_weights(source_dir, output_dir)

    tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    maybe_write_chat_template(output_dir, tokenizer)
    save_feature_extractor_and_generation_config(output_dir, args.whisper_model, source_dir)
    save_custom_config(output_dir, args, tokenizer, "stage_b")
    validate_stage_package(output_dir)


def build_stage_a_bundle(args, repo_layout_dir: Path):
    source_dir = Path(args.stage_a_dir)
    projector_path = source_dir / "audio_projector.bin"
    if not projector_path.exists():
        raise FileNotFoundError(f"Missing Stage A projector: {projector_path}")

    output_dir = repo_layout_dir / "stage_a"
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_torch_dtype(args.package_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TrainingAudioTranscriptionModel(
        whisper_model_name=args.whisper_model,
        llm_model_name=args.llm_model,
        pool_kernel=args.pool_kernel,
        torch_dtype=dtype,
    )
    prompt_enc = tokenizer(args.prompt_text, add_special_tokens=True, return_tensors="pt")
    model.set_prompt_ids(prompt_enc.input_ids.squeeze(0))
    projector_state = torch.load(projector_path, map_location="cpu", weights_only=True)
    model.audio_projector.load_state_dict(projector_state)

    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state_dict, output_dir / "model.safetensors")
    torch.save(projector_state, output_dir / "audio_projector.bin")

    tokenizer.save_pretrained(output_dir)
    maybe_write_chat_template(output_dir, tokenizer)
    save_feature_extractor_and_generation_config(output_dir, args.whisper_model, args.llm_model)
    copy_stage_metadata(source_dir, output_dir)
    save_custom_config(output_dir, args, tokenizer, "stage_a")
    validate_stage_package(output_dir)

    del state_dict
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_root_readme(output_dir: Path, args, stages: list[str]):
    stage_lines = "\n".join(f"- `{stage}`" for stage in stages)
    compare_line = 'python pull_model_and_infer.py --model-source "{repo}" --stage both --audio-path test.mp3'.format(
        repo=args.repo_id,
    )
    readme = f"""---
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- audio
- speech
- automatic-speech-recognition
- custom-code
base_model:
- {args.llm_model}
- {args.whisper_model}
---

# Audio Transcription Model Stages

This repository contains comparable exports of the same audio transcription stack at two training stages:

{stage_lines}

Each stage subfolder contains:

- the full Whisper encoder weights
- the full LLM weights
- the stage-specific `audio_projector` weights
- tokenizer and feature extractor files

## Load A Specific Stage

```python
import torch
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor

stage = "stage_b"

model = AutoModel.from_pretrained(
    "{args.repo_id}",
    subfolder=stage,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "{args.repo_id}",
    subfolder=stage,
    trust_remote_code=True,
)
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "{args.repo_id}",
    subfolder=stage,
)
```

## Compare Both Stages

```bash
{compare_line}
```
"""
    (output_dir / "README.md").write_text(readme)


def build_repo_layout(args, stages: list[str]) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model_code(output_dir)
    write_root_readme(output_dir, args, stages)

    if "stage_a" in stages:
        build_stage_a_bundle(args, output_dir)
    if "stage_b" in stages:
        build_stage_b_bundle(args, output_dir)

    return output_dir


def push_repo_layout(repo_layout_dir: Path, args):
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.repo_private, exist_ok=True)
    return api.upload_folder(
        folder_path=str(repo_layout_dir),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=None,
        commit_message=args.commit_message,
    )


def cleanup_legacy_layout(args):
    api = HfApi()
    for folder in LEGACY_STAGE_FOLDERS:
        try:
            api.delete_folder(
                folder,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Remove legacy {folder} folder",
            )
        except HfHubHTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise

    for path in LEGACY_ROOT_FILES:
        try:
            api.delete_file(
                path,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Remove legacy root file {path}",
            )
        except HfHubHTTPError as exc:
            if exc.response is None or exc.response.status_code != 404:
                raise


def main():
    args = parse_args()
    stages = parse_stages(args.stages)
    repo_layout_dir = build_repo_layout(args, stages)

    if args.skip_push:
        print(f"Packaged repo layout written to {repo_layout_dir}")
        return

    upload_info = push_repo_layout(repo_layout_dir, args)
    repo_url = getattr(upload_info, "repo_url", f"https://huggingface.co/{args.repo_id}")
    commit_url = getattr(upload_info, "commit_url", None)
    oid = getattr(upload_info, "oid", None)
    print(f"Pushed staged model packages to {args.repo_id}")
    print(f"Stages: {', '.join(stages)}")
    print(f"Repo URL: {repo_url}")
    if commit_url:
        print(f"Commit URL: {commit_url}")
    if oid:
        print(f"Commit OID: {oid}")

    if args.cleanup_legacy_layout:
        cleanup_legacy_layout(args)
        print("Legacy root-level package files removed.")


if __name__ == "__main__":
    main()
