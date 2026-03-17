import argparse
import gc

import torch
from datasets import Audio, Dataset
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor


DEFAULT_MODEL_SOURCE = "RaghaRao314159/transcription-models"
VALID_STAGES = ("stage_a", "stage_b", "both")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull one or both packaged audio transcription stages and run inference on an audio file.",
    )
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--stage", choices=VALID_STAGES, default="stage_b")
    parser.add_argument("--audio-path", default="test.mp3")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def choose_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_audio(audio_path: str, sample_rate: int = 16000):
    ds = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=sample_rate))
    decoder = ds[0]["audio"]
    samples = decoder.get_all_samples()
    waveform = samples.data
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0)
    return waveform, samples.sample_rate


def load_model_bundle(model_source: str, stage: str, revision: str | None, dtype: torch.dtype):
    common = {
        "trust_remote_code": True,
        "revision": revision,
        "subfolder": stage,
    }
    model_kwargs = {**common, "low_cpu_mem_usage": True, "dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(model_source, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_source, **common)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source, revision=revision, subfolder=stage)
    return model, tokenizer, feature_extractor


def transcribe_stage(model_source: str, stage: str, revision: str | None, dtype: torch.dtype, waveform, sample_rate: int, max_new_tokens: int) -> str:
    model, tokenizer, feature_extractor = load_model_bundle(model_source, stage, revision, dtype)
    if not hasattr(model, "device"):
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = model.device

    features = feature_extractor(
        waveform.cpu().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    ).input_features
    features = features.to(device=device, dtype=model.audio_projector[0].weight.dtype)
    audio_lengths = torch.tensor([waveform.shape[-1]], device=device)

    with torch.inference_mode():
        output_ids = model.transcribe(
            audio_features=features,
            audio_lengths=audio_lengths,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    transcript = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    del model
    del tokenizer
    del feature_extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transcript


def main():
    args = parse_args()
    dtype = choose_dtype()
    waveform, sample_rate = load_audio(args.audio_path)

    stages = ["stage_a", "stage_b"] if args.stage == "both" else [args.stage]
    transcripts = {}
    for stage in stages:
        transcripts[stage] = transcribe_stage(
            model_source=args.model_source,
            stage=stage,
            revision=args.revision,
            dtype=dtype,
            waveform=waveform,
            sample_rate=sample_rate,
            max_new_tokens=args.max_new_tokens,
        )

    if len(stages) == 1:
        print(transcripts[stages[0]])
        return

    for stage in stages:
        print(f"[{stage}]")
        print(transcripts[stage])
        print()


if __name__ == "__main__":
    main()
