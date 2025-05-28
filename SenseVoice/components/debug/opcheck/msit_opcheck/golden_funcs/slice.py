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

import tensorflow as tf

from msit_opcheck.graph_parser import OpInfo
from msit_opcheck.utils import ceil_div


def _update_params_for_other_format(shape, begin, size, input_format, ori_format):
    size_new = []
    for i, item in enumerate(size):
        if item != -1:
            size_new.append(item)
        else:
            size_new.append(shape[i] - begin[i])
    size = size_new
    align_c0 = 16
    begin = list(begin)
    size = list(size)
    if input_format in ["NDC1HWC0", "NC1HWC0", "FRACTAL_Z", "FRACTAL_Z_3D"]:
        begin_nchw = [begin[ori_format.index("N")], begin[ori_format.index("C")],
                      begin[ori_format.index("H")], begin[ori_format.index("W")]]
        size_nchw = [size[ori_format.index("N")], size[ori_format.index("C")],
                     size[ori_format.index("H")], size[ori_format.index("W")]]
        begin_c1 = begin_nchw[1] // align_c0
        begin_c0 = 0
        begin_n1 = begin_nchw[0] // align_c0
        begin_n0 = 0
        size_c1 = ceil_div(size_nchw[1], align_c0)
        size_c0 = -1
        size_n1 = ceil_div(size_nchw[0], align_c0)
        size_n0 = -1

        if input_format == "NDC1HWC0":
            begin_new = [begin_nchw[0], begin[ori_format.index("D")],
                         begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size[ori_format.index("D")],
                        size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "NC1HWC0":
            begin_new = [begin_nchw[0], begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "FRACTAL_Z_3D":
            begin_new = [begin[ori_format.index("D")],
                         begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0]
            size_new = [size[ori_format.index("D")],
                        size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]
        else:
            begin_new = [begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0]
            size_new = [size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]

        return begin_new, size_new

    else:
        begin_fisrt_last_dim_one = begin[-1] // align_c0
        begin_fisrt_last_dim_two = 0

        begin_second_last_dim_one = begin[-2] // align_c0
        begin_second_last_dim_two = 0

        size_fisrt_last_dim_one = ceil_div(size[-1], align_c0)
        size_fisrt_last_dim_two = -1

        size_second_last_dim_one = ceil_div(size[-2], align_c0)
        size_second_last_dim_two = -1

        begin_new = begin[0:-2] + [begin_fisrt_last_dim_one, begin_second_last_dim_one,
                                   begin_second_last_dim_two, begin_fisrt_last_dim_two]
        size_new = size[0:-2] + [size_fisrt_last_dim_one, size_second_last_dim_one,
                                 size_second_last_dim_two, size_fisrt_last_dim_two]

        return begin_new, size_new


def _slice(context: OpInfo):
    tf.compat.v1.disable_eager_execution()
    x_data = context.param.get("input_arrays")[0]
    begin_data = context.param.get("offsets")
    size_data = context.param.get("size")

    # format info
    input_format = context.param.get("dyn_input_formats")[0]
    ori_format = context.param.get("stc_input_ori_formats")[0]
    org_shape = context.param.get("stc_ori_inputs")[0]
    if input_format in ("NDC1HWC0", "NC1HWC0", "FRACTAL_NZ", "FRACTAL_Z", "FRACTAL_Z_3D"):
        begin_data, size_data = \
            _update_params_for_other_format(org_shape, begin_data, size_data, input_format, ori_format)

    # indices 1
    x_shape = x_data.shape
    x = tf.compat.v1.placeholder(dtype=x_data.dtype, shape=x_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.slice(x, begin_data, size_data, name=None)
        res = sess.run(gather_res, feed_dict={x: x_data, })
    return res