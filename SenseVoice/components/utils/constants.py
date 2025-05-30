# Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


AIT_FAQ_HOME = "gitee repo: Ascend/msit, wiki"
MIND_STUDIO_LOGO = "[Powered by MindStudio]"

PATH_WHITE_LIST_REGEX = re.compile(r"[^_A-Za-z0-9/.-]")

CONFIG_FILE_MAX_SIZE = 1 * 1024 * 1024 # work for .ini config file
TEXT_FILE_MAX_SIZE = 100 * 1024 * 1024 # work for txt, py
CSV_FILE_MAX_SIZE = 1024 * 1024 * 1024
JSON_FILE_MAX_SIZE = 1024 * 1024 * 1024
ONNX_MODEL_MAX_SIZE = 2 * 1024 * 1024 * 1024
TENSOR_MAX_SIZE = 10 * 1024 * 1024 * 1024
MODEL_WEIGHT_MAX_SIZE = 300 * 1024 * 1024 * 1024
INPUT_FILE_MAX_SIZE = 5 * 1024 * 1024 * 1024
LOG_FILE_MAX_SIZE = 100 * 1024 * 1024

EXT_SIZE_MAPPING = {
    ".ini": CONFIG_FILE_MAX_SIZE,
    '.csv': CSV_FILE_MAX_SIZE,
    '.json': JSON_FILE_MAX_SIZE,
    '.txt': TEXT_FILE_MAX_SIZE,
    '.py': TEXT_FILE_MAX_SIZE,
    '.pth': MODEL_WEIGHT_MAX_SIZE,
    '.bin': MODEL_WEIGHT_MAX_SIZE,
    '.onnx': ONNX_MODEL_MAX_SIZE,
}

MAX_DEPTH_LIMIT = 100
