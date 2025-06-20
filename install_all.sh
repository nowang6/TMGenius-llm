#!/bin/bash

model=""

# 检查310P3数量
count_310P3=$(npu-smi info | grep -c 310P3)
if [ "$count_310P3" -eq 8 ]; then
    model="300i-duo"
fi

# 检查910B数量
count_910B=$(npu-smi info | grep -c 910B)
if [ "$count_910B" -eq 8 ]; then
    model="800i-a2-64g"
fi

if [ -z "$model" ]; then
    echo "unknown"
    exit 1
else
    echo "机型设置为: $model"
fi

# 调用load_images.sh
bash ./load_images.sh

# 检查是否有mindie镜像
if docker images | grep -q mindie; then
    echo "已找到mindie镜像"
else
    echo "未找到mindie镜像，安装失败"
    exit 1
fi

# 执行docker compose
docker compose -f docker/docker-compose-${model}.yaml up -d
