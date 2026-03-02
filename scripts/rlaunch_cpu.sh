# 第一步：先用rlaunch申请porxy worker资源
rlaunch --charged-group=puyullm_proxy --namespace ailab-puyullmgpu --private-machine=group \
    --mount=gpfs://gpfs1/gaozhangwei:/mnt/shared-storage-user/gaozhangwei \
    --mount=gpfs://gpfs1/intern7shared:/mnt/shared-storage-user/intern7shared \
    --mount=gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin \
    --mount=gpfs://gpfs1/intern-multi-modal-delivery:/mnt/shared-storage-user/intern-multi-modal-delivery \
    --mount=gpfs://gpfs1/puyullmgpu-shared:/mnt/shared-storage-user/puyullmgpu-shared \
    --mount=gpfs://gpfs1/auto-eval-pipeline:/mnt/shared-storage-user/auto-eval-pipeline \
    --mount=gpfs://gpfs2/sfteval:/mnt/shared-storage-user/sfteval \
    --cpu 16 --memory 100000 \
    -d -- bash -c 'sleep inf'

# 第二步：ssh进worker之后，设置美国代理
# export http_proxy=http://gulixin:3U3FHVa8rcKu2aJeQdxdmKphQtuWF3bvnmlXNYct2W4iwvKU0PQCQSfzohKE@proxy-us.h.pjlab.org.cn:23128
# export https_proxy=http://gulixin:3U3FHVa8rcKu2aJeQdxdmKphQtuWF3bvnmlXNYct2W4iwvKU0PQCQSfzohKE@proxy-us.h.pjlab.org.cn:23128
# export no_proxy=10.0.0.0/8,100.96.0.0/12,.pjlab.org.cn

# 第三步：curl -I google.com 测试网络是否联通

# 第四步：设置conda环境
# export PATH=/mnt/shared-storage-user/gaozhangwei/workspace_glx/miniconda3/bin:$PATH &&
# echo $PATH &&
# source /mnt/shared-storage-user/gaozhangwei/workspace_glx/miniconda3/etc/profile.d/conda.sh &&
# conda activate vlmevalkit_mzr &&
# cd /mnt/shared-storage-user/gaozhangwei/workspace_glx/vlmevalkit_mzr
# proxy_on
