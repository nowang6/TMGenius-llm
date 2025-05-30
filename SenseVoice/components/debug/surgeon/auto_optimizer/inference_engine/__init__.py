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


__all__ = ["PreProcessBase", "PostProcessBase", "EvaluateBase", "InferenceBase"]


from auto_optimizer.inference_engine.pre_process.pre_process_base import PreProcessBase
from auto_optimizer.inference_engine.post_process.post_process_base import PostProcessBase
from auto_optimizer.inference_engine.evaluate.evaluate_base import EvaluateBase
from auto_optimizer.inference_engine.inference.inference_base import InferenceBase