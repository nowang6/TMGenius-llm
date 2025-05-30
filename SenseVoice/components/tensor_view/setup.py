# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

msit_sub_tasks = [
    {
    "name": "tensor-view",
    "help_info": "view、slice、permute、save the dumped tensor",
    "module": "ait_tensor_view.main_cli",
    "attr": "get_cmd_instance"
}
]

msit_sub_task_entry_points = [
    f"{t.get('name')}:{t.get('help_info')} = {t.get('module')}:{t.get('attr')}"
    for t in msit_sub_tasks
]

setup(
    name='msit-tensor-view',
    version='8.0.0',
    description='tensor view tool',
    long_description="Provides interfaces for viewing, slicing, transposing, and saving tensor",
    url='msit tensor-view url',
    packages=find_packages(),
    keywords='msit tensor-view tool',
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'msit_sub_task': msit_sub_task_entry_points,
        'msit_sub_task_installer': ['msit-tensor-view=ait_tensor_view.__install__:TensorViewInstaller'],
    },
)
