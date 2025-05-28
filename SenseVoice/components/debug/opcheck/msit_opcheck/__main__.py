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

from components.utils.parser import BaseCommand
from components.debug.compare.msquickcmp.common.args_check import (
    check_input_path_legality, check_output_path_legality
)

from components.debug.common import logger
from msit_opcheck.opchecker import OpChecker


class OpcheckCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = None

    def add_arguments(self, parser, **kwargs):
        parser.add_argument(
            '--input',
            '-i',
            required=True,
            type=check_input_path_legality,
            help='input directory.E.g:--input DUMP_DATA_DIR/{TIMESTAMP}')

        parser.add_argument(
            '--output',
            '-o',
            required=False,
            type=check_output_path_legality,
            default='./',
            help='Data output directory.E.g:--output /xx/xxx/xx')

    def handle(self, args):
        # do opcheck
        opchecker = OpChecker(args)
        
        opchecker.start_test()


def get_opcheck_cmd_ins():
    help_info = "Operation check tool for GE compile model."
    opcheck_instance = OpcheckCommand("opcheck", help_info)
    return opcheck_instance