# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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

import tensorflow
import numpy
import torch


DATA_TYPE_MAP = {
    "DT_FLOAT": "float32",
    "DT_FLOAT16": "float16",
    "DT_DOUBLE": "float64",
    "DT_BOOL": "bool"
}


def bfloat16_conversion(container):
    """
    Convert bfloat16 string to numpy dtype
    """
    # noinspection PyUnresolvedReferences
    return [dtype if dtype != "bfloat16" else tensorflow.bfloat16.as_numpy_dtype for dtype in container]


def bfloat16_conversion_v2(container):
    """
    Convert bfloat16/int4/fp8 string to numpy dtype
    """
    ret = list(container)
    special_dtypes = ("bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "float8_e8m0", "float4_e2m1", "float4_e1m2")
    for sd in special_dtypes:
        for idx, dtype in enumerate(ret):
            if not isinstance(dtype, str):
                continue
            if sd == dtype:
                ret[idx] = eval(f"numpy_{sd}()")
    return ret


def numpy_to_torch_tensor(np_array: numpy.ndarray, is_complex32=False):
    """
    Convert numpy data to torch.tensor
    """
    if "bfloat16" in str(np_array.dtype):
        if hasattr(torch, "frombuffer"):
            return torch.frombuffer(np_array, dtype=torch.bfloat16).reshape(np_array.shape)
        else:
            np_fp32 = np_array.astype("float32")
            t_fp32 = torch.from_numpy(np_fp32)
            return t_fp32.to(torch.bfloat16, copy=False)
    else:
        if is_complex32:
            if np_array.dtype.name == "float16":
                ret = torch.from_numpy(np_array)
                return ret.view(torch.complex32)
            else:
                raise RuntimeError(f"Can only transfer numpy.float16 to torch.complex32. Not: {np_array.dtype.name}")
        else:
            return torch.from_numpy(np_array)
