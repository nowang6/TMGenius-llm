#!/bin/bash

response=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" --location 'http://localhost:8082/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "tmgenius-agent-7b",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ]
}')

# 提取 body 和 http 状态码
body=$(echo "$response" | sed -e 's/HTTPSTATUS:.*//g')
status=$(echo "$response" | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')

if [ "$status" -eq 200 ]; then
    echo "接口响应正常，HTTP状态码: $status"
    # 检查返回内容是否包含 content 字段
    if echo "$body" | grep -q '"content"'; then
        echo "模型有返回内容："
        echo "$body"
        exit 0
    else
        echo "模型响应中未找到 content 字段，可能有异常："
        echo "$body"
        exit 1
    fi
else
    echo "接口响应异常，HTTP状态码: $status"
    echo "$body"
    exit 1
fi

