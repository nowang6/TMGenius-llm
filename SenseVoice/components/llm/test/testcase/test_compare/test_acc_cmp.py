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
import os
import shutil
import json
import stat

import pytest
from unittest import mock
import torch
import numpy as np

import msit_llm.compare.cmp_utils
from msit_llm.compare import atb_acc_cmp

from components.llm.msit_llm.common.constant import GLOBAL_AIT_DUMP_PATH

FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GOLDEN_DATA_PATH = "test_acc_cmp_fake_golden_data.npy"
FAKE_MY_DATA_PATH = "test_acc_cmp_fake_test_data.npy"


@pytest.fixture(scope="module")
def golden_data_file():
    golden_data = np.ones((2, 3)).astype(np.float32)
    golden_data[0][0] = 10
    np.save(FAKE_GOLDEN_DATA_PATH, golden_data)

    yield FAKE_GOLDEN_DATA_PATH

    if os.path.exists(FAKE_GOLDEN_DATA_PATH):
        os.remove(FAKE_GOLDEN_DATA_PATH)


@pytest.fixture(scope="module")
def test_data_file():
    test_data = np.ones((2, 3)).astype(np.float32)
    test_data[0][0] = 10
    np.save(FAKE_MY_DATA_PATH, test_data)

    yield FAKE_MY_DATA_PATH

    if os.path.exists(FAKE_MY_DATA_PATH):
        os.remove(FAKE_MY_DATA_PATH)


@pytest.fixture(scope="module")
def test_dat_path():
    test_data_path = "test_acc_cmp_fake_test_data.dat"  # No need to create actual file

    yield test_data_path

    if os.path.exists(test_data_path):
        os.remove(test_data_path)


@pytest.fixture(scope="module")
def test_metadata_path():
    test_metadata_path = "test_acc_cmp_fake_metadata"
    if not os.path.exists(test_metadata_path):
        os.makedirs(test_metadata_path, mode=0o750)
    metadata = {0: {0: [FAKE_GOLDEN_DATA_PATH, FAKE_MY_DATA_PATH]}}

    metadata_json_path = os.path.join(test_metadata_path, "metadata.json")
    with os.fdopen(os.open(metadata_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
        json.dump(metadata, ff)

    yield test_metadata_path

    if os.path.exists(test_metadata_path):
        shutil.rmtree(test_metadata_path)


@pytest.fixture(scope="module")
def test_torch_path():
    test_torch_path = "test_acc_cmp_fake_torch"
    with open("testcase/test_compare/json/model_tree.json") as f:
        torch_topo = json.load(f)

    if not os.path.exists(test_torch_path):
        os.makedirs(os.path.join(test_torch_path, "1111_npu0/0/"), mode=0o750)
        _json_path = os.path.join(test_torch_path, "1111_npu0/model_tree.json")
        with os.fdopen(os.open(_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
            json.dump(torch_topo, ff)

    yield test_torch_path

    if os.path.exists(test_torch_path):
        shutil.rmtree(test_torch_path)


@pytest.fixture(scope="module")
def test_atb_path():
    test_atb_path = "test_acc_cmp_fake_atb"
    with open("testcase/test_compare/json/Bloom7BFlashAttentionModel.json") as f:
        atb_topo = json.load(f)

    if not os.path.exists(test_atb_path):
        os.makedirs(os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/tensors/1_2222/0/"), mode=0o750)
        os.makedirs(os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/model/2222/"), mode=0o750)
        _json_path = os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/model/2222/BloomModel.json")
        with os.fdopen(os.open(_json_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
            json.dump(atb_topo, ff)

    yield test_atb_path

    if os.path.exists(test_atb_path):
        shutil.rmtree(test_atb_path)


def test_check_tensor_given_golden_data_when_nan_then_false():
    result, message = msit_llm.compare.cmp_utils.check_tensor(
        torch.zeros([2]).float() + torch.nan, torch.zeros([2]).float()
    )
    assert result is False and len(message) > 0 and "golden" in message.lower()


def test_fill_row_data_given_my_path_when_valid_then_pass(golden_data_file, test_data_file):
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, test_data_file, 0, 0)
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 22
    assert row_data["cosine_similarity"] == 1.0
    assert len(row_data["cmp_fail_reason"]) == 0


def test_fill_row_data_given_loaded_my_data_when_valid_then_pass(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data)
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 22
    assert row_data["cosine_similarity"] == 0.0


def test_fill_row_data_given_my_path_when_dir_then_error(golden_data_file):
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "/")
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 5
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_golden_data_path_when_empty_then_error(test_data_file):
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo("", test_data_file)
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info)
    assert isinstance(row_data, dict) and len(row_data) == 5
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_nan_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros_like(golden_data) + np.nan
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 15
    assert len(row_data["cmp_fail_reason"]) > 0


def test_fill_row_data_given_my_path_when_shape_not_match_then_error(golden_data_file):
    golden_data = np.load(golden_data_file)
    loaded_my_data = np.zeros([])
    data_info = msit_llm.compare.cmp_utils.BasicDataInfo(golden_data_file, "test")
    row_data = msit_llm.compare.cmp_utils.fill_row_data(data_info, loaded_my_data=loaded_my_data)
    assert isinstance(row_data, dict) and len(row_data) == 15
    assert len(row_data["cmp_fail_reason"]) > 0


def test_save_compare_reault_to_csv_given_data_frame_when_valid_then_pass():
    dd = [{"aa": 11}, {"bb": 12}]
    csv_save_path = msit_llm.compare.cmp_utils.save_compare_reault_to_csv(dd)
    assert os.path.exists(csv_save_path) and os.path.getsize(csv_save_path) > 0


def test_acc_compare_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    atb_acc_cmp.acc_compare(golden_data_file, test_data_file)


def test_read_data_given_data_file_when_valid_npy_then_pass(golden_data_file, test_data_file):
    data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden = msit_llm.compare.cmp_utils.read_data(golden_data_file)
    assert (data == golden).all()


def test_read_data_when_npy(golden_data_file, test_data_file):
    data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden = torch.tensor(np.load(golden_data_file))
    assert torch.all(data == golden).item()


def test_read_data_given_data_file_when_invalid_type_then_error(test_dat_path):
    from argparse import ArgumentError

    # check path_legality will raise FileNotFoundError instead of TypeError
    with pytest.raises(FileNotFoundError):
        msit_llm.compare.cmp_utils.read_data(test_dat_path)


def test_compare_data_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    test_data = msit_llm.compare.cmp_utils.read_data(test_data_file)
    golden_data = msit_llm.compare.cmp_utils.read_data(golden_data_file)
    res = msit_llm.compare.cmp_utils.compare_data(test_data, golden_data)
    assert res == {
        "cosine_similarity": 1.0,
        "max_relative_error": 0.0,
        "mean_relative_error": 0.0,
        "kl_divergence": 0.0,
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_euclidean_distance": 0.0,
        "cmp_fail_reason": "",
    }


def test_compare_file_given_data_file_when_valid_then_pass(golden_data_file, test_data_file):
    res = atb_acc_cmp.compare_file(golden_data_file, test_data_file)
    assert res == {
        "cosine_similarity": 1.0,
        "max_relative_error": 0.0,
        "mean_relative_error": 0.0,
        "kl_divergence": 0.0,
        "max_absolute_error": 0.0,
        "mean_absolute_error": 0.0,
        "relative_euclidean_distance": 0.0,
        "cmp_fail_reason": "",
    }


def test_compare_metadata_given_golden_path_when_valid_then_pass(test_metadata_path):
    csv_save_path = atb_acc_cmp.compare_metadata(test_metadata_path, output_path=".")
    assert os.path.exists(csv_save_path) and os.path.getsize(csv_save_path) > 0


def test_compare_torch_atb_given_data_path_when_valid_then_pass(test_torch_path, test_atb_path):
    torch_model_topo_file = os.path.join(test_torch_path, "1111_npu0/model_tree.json")
    golden_path = os.path.abspath(os.path.join(test_torch_path, "1111_npu0/0/"))
    my_path = os.path.abspath(os.path.join(test_atb_path, f"{GLOBAL_AIT_DUMP_PATH}/tensors/1_2222/0/"))
    with mock.patch("msit_llm.dump.torch_dump.topo.TreeNode.get_layer_node_type", return_value="BloomLayer"):
        csv_save_path = atb_acc_cmp.cmp_torch_atb(
            torch_model_topo_file, (golden_path, my_path, "."), mapping_file_path="."
        )
    assert os.path.exists(csv_save_path) and os.path.getsize(csv_save_path) > 0
