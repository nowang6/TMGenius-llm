#!/bin/bash

# 自动检测系统架构
ARCH=$(uname -m)
PLATFORM=""

case $ARCH in
    x86_64)
        PLATFORM="amd64"
        ;;
    aarch64)
        PLATFORM="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Detected system architecture: $ARCH, using platform: $PLATFORM"

# 登录华为云镜像仓库
#docker login -u cn-south-1@HST3UZHVUG3KOJ367GOI swr.cn-south-1.myhuaweicloud.com
# 输入密码

# 下载300I-Duo镜像
#docker pull --platform=$PLATFORM swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-300I-Duo-py311-openeuler24.03-lts

# 下载800I-A2镜像
# docker pull --platform=$PLATFORM swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-800I-A2-py311-openeuler24.03-lts

# 保存镜像
docker save -o images/mindie-1.0.0-300I-Duo-py311-openeuler24.03-lts.tar swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-300I-Duo-py311-openeuler24.03-lts
docker save -o images/mindie-1.0.0-800I-A2-py311-openeuler24.03-lts.tar swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-800I-A2-py311-openeuler24.03-lts
