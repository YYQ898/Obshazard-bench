#!/bin/bash
# 此脚本用于在集群环境中启动评测任务
# 请根据实际环境修改以下配置

# 示例：申请计算资源
# rlaunch --charged-group=YOUR_GROUP --namespace YOUR_NAMESPACE --private-machine=group \
#     --mount=gpfs://gpfs1/YOUR_PATH:/mnt/YOUR_MOUNT_POINT \
#     --cpu 16 --memory 100000 \
#     -d -- bash -c 'sleep inf'

# 设置代理（如需要）
# export http_proxy=http://YOUR_PROXY_URL
# export https_proxy=http://YOUR_PROXY_URL
# export no_proxy=10.0.0.0/8,100.96.0.0/12,.your.domain

# 激活 conda 环境
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# 运行评测
# python evaluate.py --dataset-root /path/to/dataset --raw-data-base-path /path/to/base --model-name gpt-4o
