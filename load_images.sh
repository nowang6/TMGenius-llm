#!/bin/bash

# 加载emotion-recognition镜像
# docker load -i emotion-recognition:1.0.0.tar
# 设置简短名称
# docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/emotion-recognition:1.0.0 emotion-recognition:1.0.0
# 列出加载的镜像
# docker images | grep emotion-recognition
# 加载300I-Duo镜像
docker load -i docker/images/mindie-1.0.0-300I-Duo-py311-openeuler24.03-lts.tar
# 设置简短名称
docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-300I-Duo-py311-openeuler24.03-lts mindie1.0:300i-duo

# 加载800I-A2镜像
docker load -i docker/images/mindie-1.0.0-800I-A2-py311-openeuler24.03-lts.tar
# 设置简短名称
docker tag swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-800I-A2-py311-openeuler24.03-lts mindie1.0:800i-a2


