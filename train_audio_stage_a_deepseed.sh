#!/bin/bash
# Stage A: Projector alignment (only MLP projector is trained)
# Whisper and LLM are frozen. Fast training, small memory footprint.

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
deepspeed "$SCRIPT_DIR/train_audio.py" \
    --deepspeed ./scripts/zero2.json \
    --stage a \
    --whisper_model openai/whisper-base \
    --llm_model Qwen/Qwen3-1.7B \
    --pool_kernel 4 \
    --attn_implementation sdpa \
    --max_samples_per_dataset 25000 \
    --prompt_text "Transcribe:" \
    --output_dir "$SCRIPT_DIR/checkpoints/audio_stage_a" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 1 \
    --eval_strategy steps \
    --eval_steps 5000 \
    --dataloader_num_workers 4 \
    --report_to none
