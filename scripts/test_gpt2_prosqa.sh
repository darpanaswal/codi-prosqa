#!/bin/bash
> runs/test_gpt2_prosqa.txt
source codi/bin/activate
# Test script for GPT-2 CODI on ProsQA
#
# NOTE: Update CKPT_DIR and TEST_DATA_PATH before running.

# --- PATHS (edit these) ---
CKPT_DIR="model/prosqa/codi/prosqa_gpt2_codi/gpt2/ep_20/lr_0.001/seed_11/checkpoint-2800"
TEST_DATA_PATH="data/prosqa_test.json"
# --------------------------

python -u test.py \
    --data_name "prontoqa" \
    --test_data_path "$TEST_DATA_PATH" \
    --output_dir "." \
    --model_name_or_path model/gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --batch_size 1 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "$CKPT_DIR"