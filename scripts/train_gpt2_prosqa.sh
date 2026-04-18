#!/bin/bash
> runs/train_gpt2_prosqa.txt
source codi/bin/activate

# --- PATHS ---
DATA_PATH="data/prosqa_train.json"
VAL_DATA_PATH="data/prosqa_valid.json"
SAVE_DIR="model/prosqa/codi"
# -------------

mkdir -p "$SAVE_DIR"

torchrun --nnodes 1 --nproc_per_node 1 train.py \
    --output_dir "$SAVE_DIR" \
    --expt_name prosqa_gpt2_codi \
    --logging_dir "$SAVE_DIR/logs" \
    --logging_steps 10 \
    --model_name_or_path model/gpt2 \
    --data_name prontoqa \
    --data_path "$DATA_PATH" \
    --val_data_path "$VAL_DATA_PATH" \
    --test_data_path "data/prosqa_test.json" \
    --seed 11 \
    --model_max_length 1024 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --num_train_epochs 100 \
    --learning_rate 5e-4 \
    --max_grad_norm 2.0 \
    --use_lora True \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --save_strategy "epoch" \
    --save_safetensors False \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to wandb \
    --num_latent 6 \
    --logging_strategy "steps" \
    --use_prj True \
    --prj_dim 768 \
    --prj_dropout 0.0 \
    --distill_loss_div_std True \
    --exp_mode False \
    --remove_eos True \
    --print_ref_model_stats True