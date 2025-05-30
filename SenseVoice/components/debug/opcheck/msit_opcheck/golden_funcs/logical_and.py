# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

import numpy as np

from msit_opcheck.utils import broadcast_to_maxshape
from msit_opcheck.operation_test import OperationTest
from msit_opcheck.conversion.dtype_convert import DATA_TYPE_MAP
from msit_opcheck.constants import FLOAT16


class LogicalAndOperation(OperationTest):
    def golden_calc(self, in_tensors):
        x1, x2 = in_tensors[:2]
        out_type = DATA_TYPE_MAP[self.op_param['output_desc'][0]['dtype']]
        shape_list = broadcast_to_maxshape([x1.shape, x2.shape])
        x1 = x1.astype(FLOAT16)
        x2 = x2.astype(FLOAT16)
        x1 = np.broadcast_to(x1, shape_list[-1])
        x2 = np.broadcast_to(x2, shape_list[-1])
        return [np.multiply(x1, x2).astype(out_type)]

    def test_logical_and(self):
        self.execute()
