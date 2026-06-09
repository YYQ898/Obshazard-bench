#!/bin/bash
# 配置 API 凭据（请替换为你的实际值）
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-"https://api.openai.com/v1"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"your-api-key-here"}

# 评测集根目录（请替换为你的实际路径）
DATA_ROOT=/path/to/ObsCrisis-Bench

eval_gpt4o() {
    python evaluate.py \
        --dataset-root $DATA_ROOT/$1 \
        --raw-data-base-path $DATA_ROOT \
        --target-file All.json \
        --model-name gpt-4o \
        --max-workers 64 \
        --image-max-num 500 \
        --total-max-size 47185920 \
        --temperature 0.1 \
        --max-tokens 300
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

eval_qwen3.5-397b-a17b() {
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
# eval_gpt4o cold-wave_json
# eval_gpt4o heat-wave_json
# eval_gpt4o earthquake_json
# eval_gpt4o flood_json
# eval_gpt4o "mass movement (wet)_json"
# eval_gpt4o storm_json
# eval_gpt4o "volcanic activity_json"
# eval_gpt4o wildfire_json

# eval_gemini3pro cold-wave_json
# eval_gemini3pro heat-wave_json
# eval_gemini3pro earthquake_json
# eval_gemini3pro flood_json
# eval_gemini3pro "mass movement (wet)_json"
# eval_gemini3pro storm_json
# eval_gemini3pro "volcanic activity_json"
# eval_gemini3pro wildfire_json

# eval_qwen3.5-397b-a17b cold-wave_json
# eval_qwen3.5-397b-a17b heat-wave_json
# eval_qwen3.5-397b-a17b earthquake_json
# eval_qwen3.5-397b-a17b flood_json
# eval_qwen3.5-397b-a17b "mass movement (wet)_json"
# eval_qwen3.5-397b-a17b storm_json
# eval_qwen3.5-397b-a17b "volcanic activity_json"
# eval_qwen3.5-397b-a17b wildfire_json
