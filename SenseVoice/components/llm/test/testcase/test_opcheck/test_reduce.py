import pytest
import torch
from msit_llm.opcheck.check_case.reduce import OpcheckReduceOperation, ReduceType

from mock_operation_test import MockOperationTest


OpcheckReduceOperation.__bases__ = (MockOperationTest,)

def test_golden_calc_given_reduce_sum_and_no_axis_when_valid_input_then_correct_result():
    # Arrange
    op_check = OpcheckReduceOperation()
    op_check.op_param = {'reduceType': ReduceType.REDUCE_SUM.value}
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]

    # Act
    result = op_check.golden_calc(in_tensors)

    # Assert
    assert result == [torch.tensor([10])]
