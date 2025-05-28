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
from collections import OrderedDict
import os
import stat
import shutil

import pytest
import numpy as np


from msit_llm.compare import torchair_acc_cmp


FILE_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
FAKE_GE_DUMP_DATA_NAME = "test_torchair_acc_cmp_fake_ge_dump_data"
FAKE_FX_DUMP_DATA_NAME = "test_torchair_acc_cmp_fake_fx_dump_data"
FAKE_PBTXT_FILE_NAME = torchair_acc_cmp.GE_GRAPH_FILE_PREFIX + "17065969118878_test.txt"
FAKE_PBTXT_FILE_PATH = os.path.join(FAKE_GE_DUMP_DATA_NAME, FAKE_PBTXT_FILE_NAME)


@pytest.fixture(scope="module", autouse=True)
def set_fake_parse_torchair_dump_data():
    def fake_parse_torchair_dump_data(my_path):
        return [np.ones([2, 3]).astype(np.float32)], [np.ones([2, 3]).astype(np.float32)]

    setattr(torchair_acc_cmp, "parse_torchair_dump_data", fake_parse_torchair_dump_data)


@pytest.fixture(scope="module", autouse=True)
def set_fake_set_msaccucmp_path_from_cann():
    setattr(torchair_acc_cmp, "set_msaccucmp_path_from_cann", lambda: None)


@pytest.fixture(scope="module", autouse=True)
def fake_pbtxt_file():
    contents = """
    op {
      name: "Add_2"
      output_desc {
        name: "test"
        attr {
          key: "_fx_tensor_name"
          value {
            s: "mm-aten.mm.default.OUTPUT.0"
          }
        }
        attr {
          name: "tt2"
        }
      }
    }
    op {
      name: "Cast_9"
      output_desc {
        name: "test"
        attr {
          key: "_fx_tensor_name"
          value {
            s: "mm-aten.mm.default.OUTPUT.0"
          }
        }
        attr {
          name: "tt2"
        }
      }
    }"""

    with os.fdopen(os.open(FAKE_PBTXT_FILE_PATH, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "w") as ff:
        ff.write(contents)

    yield

    if os.path.exists(FAKE_PBTXT_FILE_PATH):
        os.remove(FAKE_PBTXT_FILE_PATH)


@pytest.fixture(scope="module", autouse=True)
def fake_ge_dump_data():
    base_path = os.path.join(FAKE_GE_DUMP_DATA_NAME, "0")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "Add.Add_2.44.6.17065969121619",
        "Cast.Cast_9.19.6.17065969118878",
        "ConcatV2D.ConcatV2.42.6.17065969121611",
        "TransData.TransData_1.1.42.6.17005969121581",
        torchair_acc_cmp.FUSION_OP_TYPE + ".Add_2Cast_9ConcatV2.19.6.17065969118878",  # Fused op name
    ]
    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        with os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, FILE_PERMISSION), "wb") as ff:
            pass

    yield

    if os.path.exists(FAKE_GE_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_GE_DUMP_DATA_NAME)


@pytest.fixture(scope="module", autouse=True)
def fake_fx_dump_data():
    base_path = os.path.join(FAKE_FX_DUMP_DATA_NAME, "1/foo")
    os.makedirs(base_path, mode=0o750, exist_ok=True)

    file_names = [
        "mm-aten.mm.default.INPUT.0.20240125031118787351.npy",
        "mm-aten.mm.default.INPUT.1.20240125031118787351.npy",
        "mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy",
    ]
    for file_name in file_names:
        np.save(os.path.join(base_path, file_name), np.zeros([]))

    yield

    if os.path.exists(FAKE_FX_DUMP_DATA_NAME):
        shutil.rmtree(FAKE_FX_DUMP_DATA_NAME)


def test_get_torchair_ge_graph_path_given_path_when_valid_then_pass():
    ge_graph_path = torchair_acc_cmp.get_torchair_ge_graph_path(FAKE_GE_DUMP_DATA_NAME)
    assert ge_graph_path is not None
    assert os.path.basename(ge_graph_path[0]).startswith(torchair_acc_cmp.GE_GRAPH_FILE_PREFIX)


def test_get_torchair_ge_graph_path_given_path_when_invalid_then_none():
    ge_graph_path = torchair_acc_cmp.get_torchair_ge_graph_path(FAKE_FX_DUMP_DATA_NAME)
    assert ge_graph_path is None


def test_parse_pbtxt_to_dict_given_path_when_valid_then_pass():
    result = torchair_acc_cmp.parse_pbtxt_to_dict(FAKE_PBTXT_FILE_PATH)
    assert isinstance(result, list) and isinstance(result[0], dict)
    output_desc = {
        "name": "test",
        "attr": {"key": "_fx_tensor_name", "value": {"s": "mm-aten.mm.default.OUTPUT.0"}},
        "attr#1": {"name": "tt2"},
    }
    expected_result = [
        {"op": {"name": "Add_2", "output_desc": output_desc}},
        {"op": {"name": "Cast_9", "output_desc": output_desc}},
    ]
    assert result == expected_result


def test_init_ge_dump_data_from_bin_path_given_path_when_valid_then_pass():
    result = torchair_acc_cmp.init_ge_dump_data_from_bin_path(FAKE_GE_DUMP_DATA_NAME)
    fused_op_name = torchair_acc_cmp.FUSION_OP_TYPE + ".Add_2Cast_9ConcatV2.19.6.17065969118878"
    expected_result = [
        {
            0: {
                "Add_2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "Add.Add_2.44.6.17065969121619"),
                "Cast_9": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "Cast.Cast_9.19.6.17065969118878"),
                "ConcatV2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", "ConcatV2D.ConcatV2.42.6.17065969121611"),
                "TransData_1.1": os.path.join(
                    FAKE_GE_DUMP_DATA_NAME, "0", "TransData.TransData_1.1.42.6.17005969121581"
                ),
                "Add_2Cast_9ConcatV2": os.path.join(FAKE_GE_DUMP_DATA_NAME, "0", fused_op_name),
            }
        }
    ]
    assert result == expected_result


def test_init_fx_dump_data_from_path_given_path_when_valid_then_pass():
    result = torchair_acc_cmp.init_fx_dump_data_from_path(FAKE_FX_DUMP_DATA_NAME)
    expected_result = [
        {
            0: {
                "mm-aten.mm.default": {
                    "input": [
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.INPUT.0.20240125031118787351.npy"
                        ),
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.INPUT.1.20240125031118787351.npy"
                        ),
                    ],
                    "output": [
                        os.path.join(
                            FAKE_FX_DUMP_DATA_NAME, "1/foo", "mm-aten.mm.default.OUTPUT.0.20240125031118787351.npy"
                        )
                    ],
                }
            }
        }
    ]
    assert result == expected_result


def test_acc_compare_given_fx_when_valid_then_pass():
    csv_path = torchair_acc_cmp.acc_compare(FAKE_FX_DUMP_DATA_NAME, FAKE_GE_DUMP_DATA_NAME)
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 750  # result with matched comparing data, 284 if empty


def test_acc_compare_given_ge_with_fused_op_when_valid_then_pass():
    csv_path = torchair_acc_cmp.acc_compare(FAKE_GE_DUMP_DATA_NAME, FAKE_GE_DUMP_DATA_NAME)
    assert os.path.exists(csv_path)
    assert os.path.getsize(csv_path) > 1900  # result with mostly matched comparing data, 284 if empty


def test_sort_ge_dump_data():
    graph_map = [
        {"op": {"name": "Add_1"}},
        {"op": {"name": "Add_2"}},
        {"op": {"name": "Add_7"}},
        {"op": {"name": "Add_9"}},
        {"op": {"name": "Add_8"}},
    ]
    dump_data = {
        "Add_9": "Add_9.354.20.1818268541338513",
        "Add_2": "Add_2.355.21.1018268541338513",
        "Add_1": "Add_1.356.22.918268541338513",
        "Add_8": "Add_8.357.23.8718268541338513",
        "Add_7": "Add_7.358.24.7718268541338513",
    }
    sort_ge_dump_data = torchair_acc_cmp.sort_ge_dump_data(dump_data, graph_map)
    expected_sort_ge_dump_data = {
        "Add_1": "Add_1.356.22.918268541338513",
        "Add_2": "Add_2.355.21.1018268541338513",
        "Add_7": "Add_7.358.24.7718268541338513",
        "Add_9": "Add_9.354.20.1818268541338513",
        "Add_8": "Add_8.357.23.8718268541338513",
    }
    assert sort_ge_dump_data == OrderedDict(expected_sort_ge_dump_data)
