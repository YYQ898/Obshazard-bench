# export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_BASE_URL=http://35.220.164.252:3888/v1
export OPENAI_API_KEY=xxxxxx

eval_gpt4o() {
    python evaluate.py \
        --dataset-root /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201 \
        --raw-data-base-path /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake \
        --target-file Image_With_Station.json \
        --model-name gpt-4o \
        --max-workers 64 \
        --image-max-num 500 \
        --total-max-size 47185920 \
        --temperature 0.1 \
        --max-tokens 300
}

eval_gemini3pro() {
    python evaluate.py \
        --dataset-root /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201 \
        --raw-data-base-path /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake \
        --target-file Image_With_Station.json \
        --model-name gemini-3-pro-preview-thinking \
        --max-workers 64 \
        --image-max-num 600 \
        --total-max-size 47185920 \
        --temperature 1.0 \
        --max-tokens 65536
}

eval_qwen3.5-397b-a17b() {
    python evaluate.py \
        --dataset-root /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake0201 \
        --raw-data-base-path /mnt/shared-storage-user/intern7shared/gulixin/data/fengwu/0202/earthquake \
        --target-file Image_With_Station.json \
        --model-name qwen3.5-397b-a17b \
        --max-workers 64 \
        --image-max-num 250 \
        --total-max-size 47185920 \
        --temperature 0.6 \
        --max-tokens 65536
}

# eval_gpt4o
# eval_gemini3pro
eval_qwen3.5-397b-a17b