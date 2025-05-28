from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckFastSoftMaxGradOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckFastSoftMaxGradOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    # Assert
    mock_execute.assert_called_once()


import torch


def gen_softmax_grad(head_num, seq_len_range):
    # 实现 gen_softmax_grad 函数
    seq_len = seq_len_range[1]
    result = torch.randn(head_num * seq_len, seq_len)
    return result


@pytest.mark.parametrize("batch_size_imm, head_num_imm, seq_len_range", [
    (4, 8, (100, 300)),
    (2, 4, (50, 150)),
    (1, 2, (25, 75)),
])
def test_test_fastsoftmaxgrad_given_batch_size_head_num_seq_len_range_when_valid_input_then_execute_successfully(
        batch_size_imm, head_num_imm, seq_len_range):
    # Arrange
    op = OpcheckFastSoftMaxGradOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute') as mock_execute:
        op.test_fastsoftmaxgrad()

    # Assert
    mock_execute.assert_called_once()
