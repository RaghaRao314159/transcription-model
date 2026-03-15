#!/bin/bash
# Stage B: End-to-end fine-tuning (MLP projector + full LLM, Whisper frozen)
# Loads the trained projector from Stage A. Higher memory, lower LR.

deepspeed train_audio.py \
    --deepspeed ./scripts/zero2.json \
    --stage b \
    --pretrained_projector /workspace/checkpoints/audio_stage_a/audio_projector.bin \
    --whisper_model openai/whisper-base \
    --llm_model Qwen/Qwen3-1.7B \
    --pool_kernel 4 \
    --attn_implementation sdpa \
    --max_samples_per_dataset 25000 \
    --prompt_text "Transcribe:" \
    --output_dir /workspace/checkpoints/audio_stage_b \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none
