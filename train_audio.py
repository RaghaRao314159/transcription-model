"""
Audio Transcription Training: Whisper Encoder -> AvgPool -> MLP -> LLM

Supports any HuggingFace causal LM (e.g. Llama, Qwen) via --llm_model.

Two-stage training following the plan:
  Stage A: Train MLP projector only (Whisper + LLM frozen)
  Stage B: Train MLP projector + full LLM (Whisper frozen)

Usage:
    # Stage A (projector alignment)
    python train_audio.py \
        --stage a --llm_model meta-llama/Llama-3.2-1B \
        --output_dir ./checkpoints/stage_a \
        --bf16 True --per_device_train_batch_size 32 \
        --learning_rate 2e-3 --num_train_epochs 1

    # Stage B (full fine-tuning)
    python train_audio.py \
        --stage b --llm_model meta-llama/Llama-3.2-1B \
        --pretrained_projector ./checkpoints/stage_a/audio_projector.bin \
        --output_dir ./checkpoints/stage_b \
        --bf16 True --per_device_train_batch_size 8 \
        --learning_rate 2e-5 --gradient_checkpointing True --num_train_epochs 3
"""

import math
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperModel,
    WhisperFeatureExtractor,
    Trainer,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
WHISPER_HOP_LENGTH = 160
WHISPER_CONV_STRIDE = 2
WHISPER_SAMPLE_RATE = 16000
WHISPER_MAX_ENCODER_FRAMES = 1500


# ─── Arguments ────────────────────────────────────────────────────────────────


@dataclass
class ModelArguments:
    whisper_model: str = field(default="openai/whisper-base")
    llm_model: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "HuggingFace model ID for the causal LM (e.g. meta-llama/Llama-3.2-1B, Qwen/Qwen3-1.7B)"},
    )
    pool_kernel: int = field(default=4)
    stage: str = field(
        default="a",
        metadata={"help": "Training stage: 'a' (projector only) or 'b' (projector + LLM)"},
    )
    pretrained_projector: Optional[str] = field(
        default=None,
        metadata={"help": "Path to audio_projector.bin from Stage A"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "'flash_attention_2', 'sdpa', or None (default)"},
    )


@dataclass
class DataArguments:
    max_samples_per_dataset: int = field(default=25000)
    prompt_text: str = field(default="Transcribe:")
    hub_dataset: str = field(
        default="RaghaRao314159/transcription-model",
        metadata={"help": "HF hub mirror repo with librispeech/mls_eng configs. Tried before original sources."},
    )
    mls_dataset: str = field(default="parler-tts/mls_eng")
    librispeech_dataset: str = field(default="openslr/librispeech_asr")


@dataclass
class AudioTrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)


# ─── Model ────────────────────────────────────────────────────────────────────


class AudioTranscriptionModel(nn.Module):
    """Whisper Encoder -> AvgPool(k) -> MLP(whisper_dim -> llm_dim) -> LLM."""

    def __init__(
        self,
        whisper_model_name: str,
        llm_model_name: str,
        pool_kernel: int = 4,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.max_pooled_len = WHISPER_MAX_ENCODER_FRAMES // pool_kernel

        whisper = WhisperModel.from_pretrained(whisper_model_name, torch_dtype=torch_dtype)
        self.whisper_encoder = whisper.encoder
        del whisper.decoder
        self.whisper_encoder.requires_grad_(False)
        self.whisper_hidden = self.whisper_encoder.config.d_model

        self.avg_pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)

        llm_kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation:
            llm_kwargs["attn_implementation"] = attn_implementation
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, **llm_kwargs)
        self.llm.config.use_cache = False
        self.llm_hidden = self.llm.config.hidden_size

        self.audio_projector = nn.Sequential(
            nn.Linear(self.whisper_hidden, self.llm_hidden),
            nn.GELU(),
            nn.Linear(self.llm_hidden, self.llm_hidden),
        )

    # ── prompt management ──

    def set_prompt_ids(self, prompt_ids: torch.LongTensor):
        self.register_buffer("_prompt_ids", prompt_ids)

    # ── delegate methods so HF Trainer can call them on the outer model ──

    @property
    def config(self):
        return self.llm.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    # ── core forward ──

    def _pooled_real_len(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Number of non-padding pooled tokens per sample."""
        samples_per_frame = WHISPER_HOP_LENGTH * WHISPER_CONV_STRIDE
        encoder_frames = torch.clamp(
            (audio_lengths.float() / samples_per_frame).ceil().long(),
            max=WHISPER_MAX_ENCODER_FRAMES,
        )
        return torch.clamp(
            (encoder_frames.float() / self.pool_kernel).ceil().long(),
            max=self.max_pooled_len,
        )

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Whisper -> pool -> project.  (B, 80, 3000) -> (B, max_pooled, llm_hidden)."""
        with torch.no_grad():
            hidden = self.whisper_encoder(audio_features).last_hidden_state
        pooled = self.avg_pool(hidden.transpose(1, 2)).transpose(1, 2)
        return self.audio_projector(pooled)

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        transcript_ids: torch.Tensor,
        transcript_attention_mask: torch.Tensor,
    ):
        B = audio_features.shape[0]
        device = audio_features.device
        embed = self.llm.get_input_embeddings()

        audio_embeds = self.encode_audio(audio_features)
        pooled_lens = self._pooled_real_len(audio_lengths)

        prompt_embeds = embed(self._prompt_ids.to(device))  # (P, H)
        P = prompt_embeds.shape[0]

        transcript_embeds = embed(transcript_ids)  # (B, max_T, H)
        text_lens = transcript_attention_mask.sum(dim=1)  # (B,)

        seqs: List[torch.Tensor] = []
        labs: List[torch.Tensor] = []
        for i in range(B):
            na = pooled_lens[i].item()
            nt = text_lens[i].item()
            seqs.append(torch.cat([
                prompt_embeds,
                audio_embeds[i, :na],
                transcript_embeds[i, :nt],
            ]))
            labs.append(torch.cat([
                torch.full((P + na,), IGNORE_INDEX, dtype=torch.long, device=device),
                transcript_ids[i, :nt],
            ]))

        max_len = max(s.shape[0] for s in seqs)
        pad_embeds = audio_embeds.new_zeros(B, max_len, self.llm_hidden)
        pad_labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=device)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i in range(B):
            L = seqs[i].shape[0]
            pad_embeds[i, :L] = seqs[i]
            pad_labels[i, :L] = labs[i]
            attn_mask[i, :L] = 1

        return self.llm(
            inputs_embeds=pad_embeds,
            labels=pad_labels,
            attention_mask=attn_mask,
        )

    # ── generation (inference) ──

    @torch.no_grad()
    def transcribe(
        self,
        audio_features: torch.Tensor,
        audio_lengths: torch.Tensor,
        **generate_kwargs,
    ) -> torch.LongTensor:
        B = audio_features.shape[0]
        device = audio_features.device
        embed = self.llm.get_input_embeddings()

        audio_embeds = self.encode_audio(audio_features)
        pooled_lens = self._pooled_real_len(audio_lengths)
        prompt_embeds = embed(self._prompt_ids.to(device))

        seqs: List[torch.Tensor] = []
        for i in range(B):
            na = pooled_lens[i].item()
            seqs.append(torch.cat([prompt_embeds, audio_embeds[i, :na]]))

        max_len = max(s.shape[0] for s in seqs)
        pad_embeds = audio_embeds.new_zeros(B, max_len, self.llm_hidden)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i in range(B):
            L = seqs[i].shape[0]
            pad_embeds[i, :L] = seqs[i]
            attn_mask[i, :L] = 1

        self.llm.config.use_cache = True
        out = self.llm.generate(
            inputs_embeds=pad_embeds,
            attention_mask=attn_mask,
            **generate_kwargs,
        )
        self.llm.config.use_cache = False
        return out


# ─── Dataset / Collator ──────────────────────────────────────────────────────


class AudioTranscriptionDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor, tokenizer):
        self.data = hf_dataset
        self.feat_ext = feature_extractor
        self.tokenizer = tokenizer
        eos = tokenizer.eos_token_id
        self.eos_id = eos[0] if isinstance(eos, list) else eos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        audio = sample["audio"]
        assert audio["sampling_rate"] == WHISPER_SAMPLE_RATE

        mel = self.feat_ext(
            audio["array"], sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt",
        )
        audio_features = mel.input_features.squeeze(0)  # (80, 3000)

        text = sample["text"].strip().lower()
        ids = self.tokenizer(text, add_special_tokens=False).input_ids + [self.eos_id]

        return dict(
            audio_features=audio_features,
            audio_lengths=torch.tensor(len(audio["array"]), dtype=torch.long),
            transcript_ids=torch.tensor(ids, dtype=torch.long),
        )


@dataclass
class AudioDataCollator:
    pad_id: int

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        audio_features = torch.stack([x["audio_features"] for x in instances])
        audio_lengths = torch.stack([x["audio_lengths"] for x in instances])

        ids_list = [x["transcript_ids"] for x in instances]
        transcript_ids = nn.utils.rnn.pad_sequence(
            ids_list, batch_first=True, padding_value=self.pad_id,
        )
        transcript_attention_mask = nn.utils.rnn.pad_sequence(
            [torch.ones(len(t), dtype=torch.long) for t in ids_list],
            batch_first=True,
            padding_value=0,
        )

        return dict(
            audio_features=audio_features,
            audio_lengths=audio_lengths,
            transcript_ids=transcript_ids,
            transcript_attention_mask=transcript_attention_mask,
        )


# ─── Data preparation ────────────────────────────────────────────────────────


class _force_offline:
    """Context manager that forces fully-offline mode in datasets + huggingface_hub."""

    def __enter__(self):
        import datasets.config as _dc
        from huggingface_hub import constants as _hc
        self._old_ds = _dc.HF_DATASETS_OFFLINE
        self._old_hub = _hc.HF_HUB_OFFLINE
        _dc.HF_DATASETS_OFFLINE = True
        _hc.HF_HUB_OFFLINE = True
        return self

    def __exit__(self, *exc):
        import datasets.config as _dc
        from huggingface_hub import constants as _hc
        _dc.HF_DATASETS_OFFLINE = self._old_ds
        _hc.HF_HUB_OFFLINE = self._old_hub


def _load_librispeech(data_args: DataArguments, n: int, split: str = "train"):
    """Load LibriSpeech with fallback: cache -> hub mirror -> original source."""
    # Map split names for the original source (LibriSpeech uses train.100)
    orig_split = f"train.100[:{n}]" if split == "train" else split
    hub_split = f"{split}[:{n}]" if split == "train" else split

    # 1) Try cache (fully offline)
    try:
        logger.info(f"Loading LibriSpeech ({split}) from cache ...")
        with _force_offline():
            ls = load_dataset(data_args.librispeech_dataset, "clean", split=orig_split)
        logger.info(f"LibriSpeech ({split}) loaded from cache ({len(ls)} samples)")
        return ls
    except Exception:
        logger.info(f"LibriSpeech ({split}) not in cache.")

    # 2) Try hub mirror
    if data_args.hub_dataset:
        try:
            logger.info(f"Trying hub mirror: {data_args.hub_dataset} (librispeech, {split}) ...")
            ls = load_dataset(data_args.hub_dataset, "librispeech", split=hub_split)
            logger.info(f"LibriSpeech ({split}) loaded from hub mirror ({len(ls)} samples)")
            return ls
        except Exception as e:
            logger.warning(f"Hub mirror failed for LibriSpeech ({split}): {e}")

    # 3) Original source
    logger.info(f"Downloading LibriSpeech ({split}) from original source: {data_args.librispeech_dataset}")
    return load_dataset(data_args.librispeech_dataset, "clean", split=orig_split)


def _load_mls(data_args: DataArguments, n: int, split: str = "train"):
    """Load MLS English with fallback: local disk cache -> hub mirror -> original source."""
    cache_split = f"{split}[:{n}]" if split == "train" else split
    hub_split = f"{split}[:{n}]" if split == "train" else split

    # 1) Try local save_to_disk cache (only for train; val/test are small)
    if split == "train":
        mls_local_path = os.path.join(
            os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")),
            "mls_eng_local",
        )
        if os.path.isdir(mls_local_path):
            try:
                logger.info(f"Loading MLS ({split}) from local cache: {mls_local_path}")
                mls = load_from_disk(mls_local_path)
                mls = mls.select(range(min(n, len(mls))))
                logger.info(f"MLS ({split}) loaded from local cache ({len(mls)} samples)")
                return mls
            except Exception:
                logger.info(f"MLS ({split}) local cache load failed.")

    # 2) Try HF cache (fully offline)
    try:
        logger.info(f"Loading MLS ({split}) from HF cache ...")
        with _force_offline():
            mls = load_dataset(data_args.mls_dataset, split=cache_split, trust_remote_code=True)
        logger.info(f"MLS ({split}) loaded from cache ({len(mls)} samples)")
        return mls
    except Exception:
        logger.info(f"MLS ({split}) not in cache.")

    # 3) Try hub mirror
    if data_args.hub_dataset:
        try:
            logger.info(f"Trying hub mirror: {data_args.hub_dataset} (mls_eng, {split}) ...")
            mls = load_dataset(data_args.hub_dataset, "mls_eng", split=hub_split)
            logger.info(f"MLS ({split}) loaded from hub mirror ({len(mls)} samples)")
            return mls
        except Exception as e:
            logger.warning(f"Hub mirror failed for MLS ({split}): {e}")

    # 4) Original source
    logger.info(f"Downloading MLS ({split}) from original source: {data_args.mls_dataset}")
    return load_dataset(data_args.mls_dataset, split=cache_split, trust_remote_code=True)


def _normalize_split(ds, split_name):
    """Shuffle, normalise column names, keep only audio+text."""
    ds = ds.shuffle(seed=42)
    if "transcript" in ds.column_names:
        ds = ds.rename_column("transcript", "text")
    ds = ds.select_columns(["audio", "text"])
    ds = ds.cast_column("audio", Audio(sampling_rate=WHISPER_SAMPLE_RATE))
    return ds


def prepare_dataset(data_args: DataArguments):
    """Load LibriSpeech + MLS with fallback: cache -> hub mirror -> original source.

    Returns (train_dataset, eval_dataset) where eval_dataset is the
    validation split (None if no validation split is available).
    """
    n = data_args.max_samples_per_dataset

    # ── train ──
    ls_train = _normalize_split(_load_librispeech(data_args, n, split="train"), "train")
    mls_train = _normalize_split(_load_mls(data_args, n, split="train"), "train")
    train_combined = concatenate_datasets([ls_train, mls_train]).shuffle(seed=42)
    train_combined = train_combined.cast_column("audio", Audio(sampling_rate=WHISPER_SAMPLE_RATE))
    logger.info(f"Combined train dataset: {len(train_combined)} samples")

    # ── validation (best-effort) ──
    eval_combined = None
    try:
        ls_val = _normalize_split(_load_librispeech(data_args, n, split="validation"), "validation")
        mls_val = _normalize_split(_load_mls(data_args, n, split="validation"), "validation")
        eval_combined = concatenate_datasets([ls_val, mls_val]).shuffle(seed=42)
        eval_combined = eval_combined.cast_column("audio", Audio(sampling_rate=WHISPER_SAMPLE_RATE))
        logger.info(f"Combined validation dataset: {len(eval_combined)} samples")
    except Exception as e:
        logger.warning(f"Could not load validation split, training without eval: {e}")

    return train_combined, eval_combined


# ─── Projector save helper (handles DeepSpeed ZeRO-3 parameter gathering) ─────


def gather_projector_state_dict(model):
    state = {}
    for name, param in model.named_parameters():
        if "audio_projector" not in name:
            continue
        key = name.replace("audio_projector.", "")
        if hasattr(param, "ds_id"):
            from deepspeed import zero
            with zero.GatheredParameters([param]):
                state[key] = param.data.detach().cpu().clone()
        else:
            state[key] = param.detach().cpu().clone()
    return state


# ─── Custom Trainer ───────────────────────────────────────────────────────────


class AudioTrainer(Trainer):
    """Saves only the projector weights at checkpoints during Stage A."""

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "save_projector_only", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            ckpt_dir = os.path.join(
                self._get_output_dir(trial=trial),
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            proj_state = gather_projector_state_dict(model)
            if self.args.local_rank in (0, -1):
                torch.save(proj_state, os.path.join(ckpt_dir, "audio_projector.bin"))
            self.state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
        else:
            super()._save_checkpoint(model, trial, metrics)


# ─── Main ─────────────────────────────────────────────────────────────────────


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, AudioTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        level=logging.INFO,
    )
    local_rank = training_args.local_rank

    def rank0_print(*args):
        if local_rank in (0, -1):
            print(*args)

    torch_dtype = (
        torch.bfloat16 if training_args.bf16
        else torch.float16 if training_args.fp16
        else torch.float32
    )

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── feature extractor ──
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    # ── model ──
    rank0_print(f"Building model  whisper={model_args.whisper_model}  "
                f"llm={model_args.llm_model}  pool_kernel={model_args.pool_kernel}")
    model = AudioTranscriptionModel(
        whisper_model_name=model_args.whisper_model,
        llm_model_name=model_args.llm_model,
        pool_kernel=model_args.pool_kernel,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    prompt_enc = tokenizer(
        data_args.prompt_text, add_special_tokens=True, return_tensors="pt",
    )
    model.set_prompt_ids(prompt_enc.input_ids.squeeze(0))
    rank0_print(f"Prompt ({len(model._prompt_ids)} tokens): "
                f"{tokenizer.decode(model._prompt_ids)}")

    # ── load Stage A projector for Stage B ──
    if model_args.pretrained_projector is not None:
        path = model_args.pretrained_projector
        if os.path.isdir(path):
            path = os.path.join(path, "audio_projector.bin")
        rank0_print(f"Loading pretrained projector from {path}")
        model.audio_projector.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    # ── freeze strategy ──
    if model_args.stage == "a":
        rank0_print("Stage A: training audio_projector only")
        model.requires_grad_(False)
        for p in model.audio_projector.parameters():
            p.requires_grad = True
        training_args.save_projector_only = True
    elif model_args.stage == "b":
        rank0_print("Stage B: training audio_projector + LLM (Whisper frozen)")
        model.whisper_encoder.requires_grad_(False)
        model.audio_projector.requires_grad_(True)
        model.llm.requires_grad_(True)
        training_args.save_projector_only = False
    else:
        raise ValueError(f"Unknown stage: {model_args.stage!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    rank0_print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # ── data ──
    rank0_print("Preparing dataset ...")
    hf_train, hf_eval = prepare_dataset(data_args)
    train_dataset = AudioTranscriptionDataset(hf_train, feature_extractor, tokenizer)
    eval_dataset = (
        AudioTranscriptionDataset(hf_eval, feature_extractor, tokenizer)
        if hf_eval is not None else None
    )
    collator = AudioDataCollator(pad_id=tokenizer.pad_token_id)

    # ── trainer ──
    trainer = AudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # ── train ──
    trainer.train()
    trainer.save_state()

    # ── final save ──
    if model_args.stage == "a":
        os.makedirs(training_args.output_dir, exist_ok=True)
        proj_state = gather_projector_state_dict(model)
        if local_rank in (0, -1):
            save_path = os.path.join(training_args.output_dir, "audio_projector.bin")
            torch.save(proj_state, save_path)
            rank0_print(f"Projector saved to {save_path}")
    else:
        trainer.save_model()
        rank0_print(f"Full model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
