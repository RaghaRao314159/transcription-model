# Hybrid Audio-to-LLM Transcription

> Research-oriented speech transcription system that bridges a frozen Whisper encoder to a causal LLM through pooled audio tokens and a trainable projector.

`transcript = LLM([prompt ; projector(avgpool(whisper(audio)))])`

## At a Glance

| Axis | Implementation |
| --- | --- |
| Task | Autoregressive speech-to-text transcription |
| Acoustic encoder | `openai/whisper-base` |
| Sequence compression | `AvgPool1d(kernel=4, stride=4)` |
| Cross-modal bridge | 2-layer MLP audio projector |
| Decoder | Hugging Face causal LM interface; current runs use `Qwen/Qwen3-1.7B` |
| Data | 50,000 training samples = 25k LibriSpeech + 25k MLS English |
| Curriculum | Stage A projector alignment -> Stage B end-to-end LLM adaptation |
| Packaging | Custom Hugging Face model export with `stage_a` and `stage_b` bundles |

The planning document started from a Llama-based design in [`plan.md`](plan.md); the executable training and export scripts in this repository currently instantiate the same recipe with `Qwen/Qwen3-1.7B`.

## System Architecture

```mermaid
flowchart LR
    A[Waveform<br/>16 kHz] --> B[Whisper feature extractor]
    B --> C[Frozen Whisper encoder]
    C --> D[1500 acoustic frames]
    D --> E[Average pooling<br/>kernel=4 stride=4]
    E --> F[<=375 pooled audio tokens]
    F --> G[2-layer MLP projector]
    H[Prompt: Transcribe:] --> I[Prompt embeddings]
    G --> J[Concatenate prompt and audio tokens]
    I --> J
    J --> K[Causal LLM decoder<br/>Qwen3-1.7B in current runs]
    K --> L[Autoregressive transcript]
```

## Data and Training Pipeline

```mermaid
flowchart LR
    A[LibriSpeech clean] --> C[Mirror or cache fallback]
    B[MLS English] --> C
    C --> D[Sample 25k from each source]
    D --> E[Shuffle and unify schema]
    E --> F[Cast audio to 16 kHz]
    F --> G[Whisper log-Mel features]
    E --> H[Lowercase transcript]
    H --> I[LLM tokenization plus EOS]
    G --> J[Audio-text batches]
    I --> J
    K[split_and_push.py] --> L[Validation and test splits]
    L --> J
```

## Two-Stage Optimization Strategy

```mermaid
flowchart TD
    A[Stage A<br/>Frozen Whisper<br/>Frozen LLM] --> B[Train projector only]
    B --> C[Export audio_projector.bin]
    C --> D[Stage B initialization]
    D --> E[Keep Whisper frozen]
    E --> F[Train projector plus full LLM]
    F --> G[Export full model checkpoint]
```

## Experimental Snapshot

| Stage | Trainable blocks | Epochs | Steps | Runtime | Train loss | Val loss |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Stage A | Projector only | 1 | 1532 | 31.8 min | `7.039 -> 0.275` | `4.165 -> 0.320` |
| Stage B | Projector + Qwen | 3 | 2298 | 55.3 min | `0.389 -> 0.253` | `0.318 -> 0.265` |

| Stage A dynamics | Stage B dynamics |
| --- | --- |
| ![Stage A loss curve](checkpoints/audio_stage_a/loss_curves.png) | ![Stage B loss curve](checkpoints/audio_stage_b/loss_curves.png) |

Current committed quantitative evidence in the repository is training and validation loss; a standalone WER benchmark script is not yet checked in.

## Packaging and Inference Path

```mermaid
flowchart LR
    A[checkpoints/audio_stage_a] --> C[push_model_huggingface.py]
    B[checkpoints/audio_stage_b] --> C
    C --> D[artifacts/hf_audio_models/stage_a]
    C --> E[artifacts/hf_audio_models/stage_b]
    D --> F[Hugging Face model repo]
    E --> F
    G[test.mp3 or local audio] --> H[pull_model_and_infer.py]
    F --> H
    H --> I[Stage-specific transcription]
```

## Repository Map

| Path | Role |
| --- | --- |
| `plan.md` | Original research and implementation plan |
| `train_audio.py` | Core training loop, dataset preparation, stage control, loss plotting |
| `train_audio_stage_a.sh` | Stage A launcher |
| `train_audio_stage_b.sh` | Stage B launcher |
| `split_and_push.py` | Creates validation and test splits for the mirrored datasets |
| `push_model_huggingface.py` | Packages checkpoints into reusable Hugging Face bundles |
| `pull_model_and_infer.py` | Pulls packaged models and runs transcription on local audio |
| `audio_transcription_config.py` | Custom Hugging Face config for exported models |
| `audio_transcription_model.py` | Custom Hugging Face model class for exported models |

## Operational Entry Points

| Goal | Command |
| --- | --- |
| Train Stage A | `bash train_audio_stage_a.sh` |
| Train Stage B | `bash train_audio_stage_b.sh` |
| Create validation and test splits | `python split_and_push.py` |
| Package model stages | `python push_model_huggingface.py` |
| Compare Stage A vs Stage B inference | `python pull_model_and_infer.py --stage both --audio-path test.mp3` |
