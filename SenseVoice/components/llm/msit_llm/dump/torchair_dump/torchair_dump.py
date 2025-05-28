# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import time

from msit_llm.common.log import logger
from msit_llm.common.utils import check_output_path_legality
from components.utils.security_check import ms_makedirs


def try_import_torchair():
    try:
        import torch
        import torch_npu
        import torchair
    except ModuleNotFoundError as ee:
        logger.error("torch or torch_npu with torchair not found. Try installing the latest torch_npu.")
        raise ee


def get_ge_dump_config(dump_path="msit_ge_dump", dump_mode="all", fusion_switch_file=None,
                       dump_token=None, dump_layer=None):
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    check_output_path_legality(dump_path)
    dump_path = os.path.join(dump_path, "dump_" + time.strftime('%Y%m%d_%H%M%S'))  # Timestamp like '20240222_095519'
    if not os.path.exists(dump_path):
        ms_makedirs(dump_path, mode=0o750)

    # Generate GE mapping graph
    config.debug.graph_dump.type = "txt"
    if hasattr(config.debug.graph_dump, "_path"):  # interface changed since 8.0.RC1.b080
        setattr(config.debug.graph_dump, "_path", dump_path)
    else:
        config.debug.graph_dump.path = dump_path

    if fusion_switch_file is not None:
        if not os.path.exists(fusion_switch_file):
            raise FileNotFoundError(f'fusion_switch_file: {fusion_switch_file} not found')
        config.fusion_config.fusion_switch_file = fusion_switch_file

    # Enable GE dump
    config.dump_config.enable_dump = True
    config.dump_config.dump_mode = dump_mode
    config.dump_config.dump_path = dump_path

    if dump_token is not None:
        new_token = [str(x) for x in dump_token]
        config.dump_config.dump_step = "|".join(new_token)
    if dump_layer is not None:
        dump_layer = " ".join(dump_layer)
        config.dump_config.dump_layer = dump_layer

    return config


def get_fx_dump_config():
    try_import_torchair()

    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    # Enable FX dump
    config.debug.data_dump.type = "npy"
    return config
