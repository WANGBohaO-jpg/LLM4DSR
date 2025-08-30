#!/bin/bash

# /c23034/wbh/code/Noise_LLM4Rec/final_denoise.sh /c23034/wbh/code/Noise_LLM4Rec/save_denoise_model/amazon_game_noise_user0.0/batch128_sample5000_epoch50/checkpoint-600 /c23034/wbh/code/Noise_LLM4Rec/data/amazon_game_noise_user0.0 prob
# /c23034/wbh/code/Noise_LLM4Rec/final_denoise.sh /c23034/wbh/code/Noise_LLM4Rec/save_denoise_model/amazon_game_noise_user0.0/batch128_sample5000_epoch50/checkpoint-600 /c23034/wbh/code/Noise_LLM4Rec/data/amazon_game_noise_user0.0 ppl

DENOISE_MODEL_PATH=$1
NOISE_DATASET_PATH=$2

SAMPLE_NUMBER=$(echo "$DENOISE_MODEL_PATH" | grep -oP "_sample\K\d+" || echo "")
CHECKPOINT_NUM=$(echo "$DENOISE_MODEL_PATH" | grep -oP "checkpoint-\K\d+" || echo "")
DENOISE_DATASET_PATH=$(echo "$NOISE_DATASET_PATH" | sed 's/noise/denoise/')
DENOISE_DATASET_PATH="${DENOISE_DATASET_PATH}_${SAMPLE_NUMBER}"
if [[ -n "$CHECKPOINT_NUM" ]]; then
    DENOISE_DATASET_PATH="${DENOISE_DATASET_PATH}_${CHECKPOINT_NUM}"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch generate_noise_prob_data.py \
    --denoise_model_path $DENOISE_MODEL_PATH \
    --noise_dataset_path $NOISE_DATASET_PATH \
    --denoise_data train --sample -1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch generate_noise_prob_data.py \
    --denoise_model_path $DENOISE_MODEL_PATH \
    --noise_dataset_path $NOISE_DATASET_PATH \
    --denoise_data test --sample 5000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch generate_suggested_data.py \
    --denoise_model_path $DENOISE_MODEL_PATH \
    --denoise_dataset_path $DENOISE_DATASET_PATH

CUDA_VISIBLE_DEVICES=0 python generate_suggested_ids.py --data_path $DENOISE_DATASET_PATH
