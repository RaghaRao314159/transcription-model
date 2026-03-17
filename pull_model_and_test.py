import argparse
import gc

import torch
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor


DEFAULT_MODEL_SOURCE = "RaghaRao314159/transcription-models"
DEFAULT_DATASET_SOURCE = "RaghaRao314159/transcription-dataset"
DEFAULT_DATASET_CONFIGS = ("librispeech", "mls_eng")
VALID_STAGES = ("stage_a", "stage_b", "both")
WHISPER_SAMPLE_RATE = 16000


class _force_offline:
    """Context manager that forces fully-offline mode in datasets + huggingface_hub."""

    def __enter__(self):
        import datasets.config as datasets_config
        from huggingface_hub import constants as hub_constants

        self._old_ds = datasets_config.HF_DATASETS_OFFLINE
        self._old_hub = hub_constants.HF_HUB_OFFLINE
        datasets_config.HF_DATASETS_OFFLINE = True
        hub_constants.HF_HUB_OFFLINE = True
        return self

    def __exit__(self, *exc):
        import datasets.config as datasets_config
        from huggingface_hub import constants as hub_constants

        datasets_config.HF_DATASETS_OFFLINE = self._old_ds
        hub_constants.HF_HUB_OFFLINE = self._old_hub


def parse_dataset_configs(raw_value: str) -> list[str]:
    configs = []
    for raw_config in raw_value.split(","):
        config = raw_config.strip()
        if config and config not in configs:
            configs.append(config)
    return configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull one or both packaged audio transcription stages and evaluate them on the combined test split using word error rate.",
    )
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--hub-dataset", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument(
        "--dataset-configs",
        default=",".join(DEFAULT_DATASET_CONFIGS),
        help="Comma-separated dataset configs to combine from the hub test split.",
    )
    parser.add_argument("--stage", choices=VALID_STAGES, default="both")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer.")

    args.dataset_configs = parse_dataset_configs(args.dataset_configs)
    if not args.dataset_configs:
        parser.error("--dataset-configs must contain at least one dataset config.")

    return args


def choose_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_bundle(model_source: str, stage: str, revision: str | None, dtype: torch.dtype):
    print(f"Loading {stage} model from {model_source} (cache first)...")
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
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_source,
        revision=revision,
        subfolder=stage,
    )
    return model, tokenizer, feature_extractor


def load_test_config(dataset_source: str, config_name: str):
    print(f"Loading {config_name}:test from local cache...")
    try:
        with _force_offline():
            dataset = load_dataset(dataset_source, config_name, split="test")
        return dataset, "local cache"
    except Exception:
        print(f"{config_name}:test not found locally, downloading from {dataset_source}...")
        dataset = load_dataset(dataset_source, config_name, split="test")
        return dataset, "huggingface"


def normalize_split(ds):
    if "text" not in ds.column_names and "transcript" in ds.column_names:
        ds = ds.rename_column("transcript", "text")

    required_columns = {"audio", "text"}
    missing_columns = required_columns.difference(ds.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Dataset split is missing required columns: {missing}")

    ds = ds.select_columns(["audio", "text"])
    ds = ds.cast_column("audio", Audio(sampling_rate=WHISPER_SAMPLE_RATE))
    return ds


def load_test_dataset(args):
    config_datasets = []

    for config_name in args.dataset_configs:
        dataset, source = load_test_config(args.hub_dataset, config_name)
        dataset = normalize_split(dataset)
        print(f"Loaded {len(dataset)} test samples from {config_name} via {source}")
        config_datasets.append(dataset)

    dataset = config_datasets[0] if len(config_datasets) == 1 else concatenate_datasets(config_datasets)
    dataset = dataset.shuffle(seed=args.shuffle_seed)
    full_size = len(dataset)

    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, full_size)))

    return dataset, full_size


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def word_edit_distance(reference_words: list[str], hypothesis_words: list[str]) -> int:
    if not reference_words:
        return len(hypothesis_words)
    if not hypothesis_words:
        return len(reference_words)

    previous = list(range(len(hypothesis_words) + 1))
    for i, ref_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def extract_waveform(audio_value):
    if isinstance(audio_value, dict):
        return torch.as_tensor(audio_value["array"]), audio_value["sampling_rate"]

    if hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        return torch.as_tensor(samples.data), samples.sample_rate

    raise TypeError(f"Unsupported audio sample type: {type(audio_value)!r}")


def transcribe_sample(model, tokenizer, feature_extractor, waveform, sample_rate: int, max_new_tokens: int) -> str:
    device = getattr(model, "device", next(model.parameters()).device)
    features = feature_extractor(
        waveform.cpu().float().numpy(),
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

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def evaluate_stage(model_source: str, stage: str, revision: str | None, dtype: torch.dtype, dataset, max_new_tokens: int):
    model, tokenizer, feature_extractor = load_model_bundle(model_source, stage, revision, dtype)
    try:
        if not hasattr(model, "hf_device_map"):
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        total_errors = 0
        total_reference_words = 0
        sample_count = len(dataset)

        for idx, sample in enumerate(dataset, start=1):
            waveform, sample_rate = extract_waveform(sample["audio"])
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)

            prediction = normalize_text(
                transcribe_sample(
                    model=model,
                    tokenizer=tokenizer,
                    feature_extractor=feature_extractor,
                    waveform=waveform,
                    sample_rate=sample_rate,
                    max_new_tokens=max_new_tokens,
                )
            )
            reference = normalize_text(sample["text"])
            reference_words = reference.split()
            total_errors += word_edit_distance(reference_words, prediction.split())
            total_reference_words += len(reference_words)

            if idx == sample_count or idx % 25 == 0:
                print(f"[{stage}] processed {idx}/{sample_count} samples")

        wer = 0.0 if total_reference_words == 0 else total_errors / total_reference_words
        return {
            "samples": sample_count,
            "word_errors": total_errors,
            "reference_words": total_reference_words,
            "wer": wer,
        }
    finally:
        del model
        del tokenizer
        del feature_extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    dtype = choose_dtype()
    dataset, full_size = load_test_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("Resolved test dataset is empty.")

    if args.num_samples is None:
        print(f"Evaluating the full combined test set: {full_size} samples")
    else:
        print(f"Evaluating {len(dataset)} of {full_size} combined test samples")

    stages = ["stage_a", "stage_b"] if args.stage == "both" else [args.stage]
    for stage in stages:
        metrics = evaluate_stage(
            model_source=args.model_source,
            stage=stage,
            revision=args.revision,
            dtype=dtype,
            dataset=dataset,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"[{stage}]")
        print(f"  samples={metrics['samples']}")
        print(f"  wer={metrics['wer']:.6f}")
        print(f"  word_errors={metrics['word_errors']}")
        print(f"  reference_words={metrics['reference_words']}")
        print()


if __name__ == "__main__":
    main()
