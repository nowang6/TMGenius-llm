# Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
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

from __future__ import print_function

import datetime
import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
import torch

from msit_llm.compare.cmp_utils import BasicDataInfo
from msit_llm.common.constant import (TOKEN_ID, DATA_ID, GOLDEN_DATA_PATH, MY_DATA_PATH,
                                      CMP_FAIL_REASON, GOLDEN_DTYPE, GOLDEN_SHAPE,
                                      GOLDEN_MAX_VALUE, GOLDEN_MIN_VALUE,
                                      GOLDEN_MEAN_VALUE, MY_DTYPE, MY_SHAPE,
                                      MY_MAX_VALUE, MY_MIN_VALUE, MY_MEAN_VALUE,
                                      CSV_GOLDEN_HEADER, GLOBAL_HISTORY_AIT_DUMP_PATH_LIST)
from msit_llm.compare.cmp_utils import (fill_row_data, load_as_torch_tensor,
                                         set_tensor_basic_info_in_row_data, compare_data, 
                                         read_data, save_compare_reault_to_csv)


@pytest.fixture(scope='module', autouse=True)
def golden_data_path():
    golden_data_path = "msit_dump_20240101_000000/torch_tensors/npu0_11111/1"
    yield golden_data_path


@pytest.fixture(scope='module', autouse=True)
def my_data_path():
    my_data_path = "msit_dump_20240101_000000/tensors/0_22222/2"
    yield my_data_path


@pytest.fixture(scope='module', autouse=True)
def sub_path():
    sub_path = "3_Prefill_layer/0_Attention/3_SelfAttention"
    yield sub_path


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path1():
    unvalid_data_path1 = "tensor_dump/tensors/0_22222/2"
    yield unvalid_data_path1


@pytest.fixture(scope='module', autouse=True)
def unvalid_data_path2():
    unvalid_data_path2 = "msit_dump_20240101_000000/tensors/0_22222/not_int"
    yield unvalid_data_path2


def test_get_token_id_from_golden_data_path(golden_data_path, my_data_path):
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    assert data_info.token_id == 2


def test_get_token_id_from_golden_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    data_info = BasicDataInfo(my_data_path, golden_data_path)
    assert data_info.token_id == 1


def test_get_token_id_from_my_data_path_sub(golden_data_path, my_data_path, sub_path):
    my_data_path = os.path.join(my_data_path, sub_path)
    golden_data_path = os.path.join(golden_data_path, sub_path)
    data_info = BasicDataInfo(golden_data_path, my_data_path)
    assert data_info.token_id == 2


def test_get_token_id_unvalid1(golden_data_path, unvalid_data_path1):
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path1)
    assert data_info.token_id == 0


def test_get_token_id_unvalid2(golden_data_path, unvalid_data_path2):
    data_info = BasicDataInfo(golden_data_path, unvalid_data_path2)
    assert data_info.token_id == 0


def test_given_token_id_data_id(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path, token_id=3, data_id=4)
    assert data_info.token_id == 3
    assert data_info.data_id == 4


def test_given_data_id(golden_data_path, my_data_path):
    data_info = BasicDataInfo(golden_data_path, my_data_path, data_id=4)
    assert data_info.token_id == 2
    assert data_info.data_id == 4


def test_given_token_id(golden_data_path, my_data_path):
    BasicDataInfo.count_data_id = 0
    data_info1 = BasicDataInfo(golden_data_path, my_data_path, token_id=3)
    assert data_info1.token_id == 3
    assert data_info1.data_id == 0

    data_info1 = BasicDataInfo(golden_data_path, my_data_path, token_id=4)
    assert data_info1.token_id == 4
    assert data_info1.data_id == 1


class TestFillRowData(unittest.TestCase):
    @patch('msit_llm.common.log.logger')  
    @patch('msit_llm.compare.cmp_utils.read_data')  
    def test_fill_row_data_with_missing_golden_data_file(self, mock_read_data, mock_logger):
        # 设置模拟数据
        class MockBasicDataInfo:
            def __init__(self):
                self.golden_data_path = 'missing_golden_data.npy'
                self.my_data_path = 'my_data.npy'
            
            @staticmethod
            def to_dict():
                return {}
 
        data_info = MockBasicDataInfo()
        mock_read_data.side_effect = [None, torch.tensor(np.array([1, 2, 3], dtype=np.float32))]
 
        # 调用函数
        result = fill_row_data(data_info)
 
        # 验证
        self.assertIn(CMP_FAIL_REASON, result)
        self.assertEqual(result[CMP_FAIL_REASON], f"golden_data_path: {data_info.golden_data_path} is not a file.")
 
 
class TestLoadAsTorchTensor(unittest.TestCase):
    def test_load_as_torch_tensor_with_loaded_data(self):
        data = np.array([1, 2, 3], dtype=np.float32)
        result = load_as_torch_tensor(None, data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.float32)
 
    def test_load_as_torch_tensor_with_unsupported_dtype(self):
        data = np.array([1, 2, 3], dtype=np.int64)  # 假设int64是不支持的
        BasicDataInfo.TORCH_UNSUPPORTED_D_TYPE_MAP = {np.int64: torch.float32}
        result = load_as_torch_tensor(None, data)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.dtype, torch.int64)

 
class TestSetTensorBasicInfoInRowData(unittest.TestCase):
    def test_set_tensor_basic_info_in_row_data(self):
        golden_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        my_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = set_tensor_basic_info_in_row_data(golden_data, my_data)
 
        self.assertEqual(result[GOLDEN_DTYPE], str(torch.float32))
        self.assertEqual(result[GOLDEN_SHAPE], str([3]))
        self.assertEqual(result[GOLDEN_MAX_VALUE], 3.0)
        self.assertEqual(result[GOLDEN_MIN_VALUE], 1.0)
        self.assertEqual(result[GOLDEN_MEAN_VALUE], 2.0)
        self.assertEqual(result[MY_DTYPE], str(torch.float32))
        self.assertEqual(result[MY_SHAPE], str([3]))
        self.assertEqual(result[MY_MAX_VALUE], 3.0)
        self.assertEqual(result[MY_MIN_VALUE], 1.0)
        self.assertEqual(result[MY_MEAN_VALUE], 2.0)


# Mock logger
logger = MagicMock()
 
# Mock functions
def read_atb_data(data_path):
    # Mock function to return a tensor
    return torch.tensor([1, 2, 3])
 
def safe_torch_load(data_path, map_location=torch.device("cpu")):
    # Mock function to return a tensor
    return torch.tensor([4, 5, 6])
 
class TestFunctions(unittest.TestCase):
    
    def test_save_compare_result_to_csv(self):
        # Mock datetime
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.datetime(2023, 10, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
            
            # Prepare data
            gathered_row_data = [
                {'Column1': 'Test1', 'Column2': 'Result1', 'Golden_Dtype': 'torch.float32', 'My_Dtype': 'torch.float32'},
                {'Column1': 'Test2', 'Column2': 'Result2', 'Golden_Dtype': 'torch.int8', 'My_Dtype': 'torch.float32'}
            ]
            output_path = "./test_output"
            
            # Mock os and pandas
            with patch('os.makedirs') as mock_makedirs, \
                 patch('os.path.join', return_value="mocked_path"), \
                 patch('pandas.DataFrame.to_csv') as mock_to_csv:
                
                # Call the function
                result_path = save_compare_reault_to_csv(gathered_row_data, output_path)
                
                # Assertions
                self.assertEqual(result_path, "mocked_path")
                mock_makedirs.assert_called_once_with(output_path, exist_ok=True)
                self.assertEqual(len(gathered_row_data), 2) 
