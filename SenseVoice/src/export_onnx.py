# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from model import SenseVoiceSmall
from utils import export_utils

if __name__ == '__main__':
    model, kwargs = SenseVoiceSmall.from_pretrained(model="/home/niwang/models/SenseVoiceSmall", device="cpu")

    rebuilt_model = model.export(type="onnx", quantize=False)

    # export model
    with torch.no_grad():
        del kwargs['model']
        export_dir = export_utils.export(model=rebuilt_model, **kwargs)
        print("Export onnx to SenseVoiceSmall")
