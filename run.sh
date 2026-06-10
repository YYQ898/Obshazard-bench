#!/bin/bash
# 配置 API 凭据（请设置环境变量或在此处填写）
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1"
export OPENAI_API_KEY="sk-ARPcdL318LhQMYPogsnKd6EJn5GTuEVMAvq9ikdyF6Czs9X4"

# 评测集根目录（请替换为你的实际路径）
DATA_ROOT=./test

eval_gpt5_5() {
    python evaluate.py \
        --dataset-root $DATA_ROOT/$1 \
        --raw-data-base-path $DATA_ROOT \
        --target-file All.json \
        --model-name gpt-5.5\
        --max-workers 64 \
        --image-max-num 500 \
        --total-max-size 47185920 \
        --temperature 0.1 \
        --max-tokens 8192
}

eval_gemini3pro() {
    python evaluate.py \
        --dataset-root $DATA_ROOT/$1 \
        --raw-data-base-path $DATA_ROOT \
        --target-file All.json \
        --model-name gemini-3-pro-preview-thinking \
        --max-workers 64 \
        --image-max-num 600 \
        --total-max-size 47185920 \
        --temperature 1.0 \
        --max-tokens 65536
}

eval_qwen3_5_397b_a17b() {
    python evaluate.py \
        --dataset-root $DATA_ROOT/$1 \
        --raw-data-base-path $DATA_ROOT \
        --target-file All.json \
        --model-name qwen3.5-397b-a17b \
        --max-workers 64 \
        --image-max-num 250 \
        --total-max-size 47185920 \
        --temperature 0.6 \
        --max-tokens 65536
}

# 用法示例（取消注释对应的行即可运行）:
# eval_gpt5_5 cold-wave_json
# eval_gpt5_5 heat-wave_json
eval_gpt5_5 earthquake_json
# eval_gpt5_5 flood_json
# eval_gpt5_5 "mass movement (wet)_json"
# eval_gpt5_5 storm_json
# eval_gpt5_5 "volcanic activity_json"
# eval_gpt5_5 wildfire_json

# eval_gemini3pro cold-wave_json
# eval_gemini3pro heat-wave_json
# eval_gemini3pro earthquake_json
# eval_gemini3pro flood_json
# eval_gemini3pro "mass movement (wet)_json"
# eval_gemini3pro storm_json
# eval_gemini3pro "volcanic activity_json"
# eval_gemini3pro wildfire_json

# eval_qwen3_5_397b_a17b cold-wave_json
# eval_qwen3_5_397b_a17b heat-wave_json
# eval_qwen3_5_397b_a17b earthquake_json
# eval_qwen3_5_397b_a17b flood_json
# eval_qwen3_5_397b_a17b "mass movement (wet)_json"
# eval_qwen3_5_397b_a17b storm_json
# eval_qwen3_5_397b_a17b "volcanic activity_json"
# eval_qwen3_5_397b_a17b wildfire_json
