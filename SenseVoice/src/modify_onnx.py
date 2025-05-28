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

import argparse
from auto_optimizer import OnnxGraph


def delete_domian(graph):
    for node in graph.nodes:
        if node.domain != '':
            node.domain = ''
    while len(graph.opset_imports) > 1:
        graph.opset_imports.pop(1)

if __name__ == '__main__':
    input_path = "/home/niwang/models/SenseVoiceSmall/model.onnx"
    save_path= "/home/niwang/models/SenseVoiceSmall/model_md.onnx"
    onnx_graph = OnnxGraph.parse(input_path)
    # 删除多余的domin
    delete_domian(onnx_graph)
    onnx_graph.save(save_path)
