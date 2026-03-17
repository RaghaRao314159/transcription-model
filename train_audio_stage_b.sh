#!/bin/bash
# Stage B: End-to-end fine-tuning (MLP projector + full LLM, Whisper frozen)
# Loads the trained projector from Stage A. Higher memory, lower LR.

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
# Ensure both GPUs are visible so each process gets its own (adjust if using different GPUs)
export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use all GPUs (adjust --num_processes if needed)
accelerate launch --num_processes 2 "$SCRIPT_DIR/train_audio.py" \
    --stage b \
    --pretrained_projector "$SCRIPT_DIR/checkpoints/audio_stage_a/audio_projector.bin" \
    --whisper_model openai/whisper-base \
    --llm_model Qwen/Qwen3-1.7B \
    --pool_kernel 4 \
    --attn_implementation sdpa \
    --max_samples_per_dataset 25000 \
    --prompt_text "Transcribe:" \
    --output_dir "$SCRIPT_DIR/checkpoints/audio_stage_b" \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 20000 \
    --save_total_limit 2 \
    --eval_strategy steps \
    --eval_steps 50 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers 4 \
    --report_to none
