from unittest.mock import patch

import torch
import pytest

from msit_llm.opcheck.check_case import OpcheckAllReduceOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckAllReduceOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([4.0, 6.0])]),
    ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([6.0])]),
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([9.0, 12.0])]),
])
def test_sum_cal_given_in_tensors_when_valid_input_then_correct_result(in_tensors, expected_result):
    # Act
    result = OpcheckAllReduceOperation.sum_cal(in_tensors)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
    ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([3.0])]),
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([5.0, 6.0])]),
])
def test_max_cal_given_in_tensors_when_valid_input_then_correct_result(in_tensors, expected_result):
    # Act
    result = OpcheckAllReduceOperation.max_cal(in_tensors)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
    ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([1.0])]),
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([1.0, 2.0])]),
])
def test_min_cal_given_in_tensors_when_valid_input_then_correct_result(in_tensors, expected_result):
    # Act
    result = OpcheckAllReduceOperation.min_cal(in_tensors)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("in_tensors, expected_result", [
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 8.0])]),
    ([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])], [torch.tensor([6.0])]),
    ([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])], [torch.tensor([15.0, 48.0])]),
])
def test_prod_cal_given_in_tensors_when_valid_input_then_correct_result(in_tensors, expected_result):
    # Act
    result = OpcheckAllReduceOperation.prod_cal(in_tensors)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("all_reduce_type, in_tensors, expected_result", [
    ("sum", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([4.0, 6.0])]),
    ("max", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])]),
    ("min", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 2.0])]),
    ("prod", [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 8.0])]),
])
def test_golden_calc_given_all_reduce_type_in_tensors_when_valid_input_then_correct_result(all_reduce_type, in_tensors,
                                                                                           expected_result):
    # Arrange
    op = OpcheckAllReduceOperation()
    op.op_param = {'allReduceType': all_reduce_type}

    def get_new_in_tensors():
        return in_tensors

    op.get_new_in_tensors = get_new_in_tensors

    # Act
    result = op.golden_calc(None)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("pid, op_param, validate_param_return, expected_execute_call", [
    (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, True),
    (None, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, False),
    (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
    (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
    (1, {'allReduceType': 'sum', 'rankRoot': 0, 'rankSize': 2}, False, False),
    (1, {'allReduceType': 'sum', 'rank': 0, 'rankSize': 2}, False, False),
    (1, {'allReduceType': 'sum', 'rank': 0, 'rankRoot': 0}, False, False),
])
def test_test_all_reduce_given_pid_op_param_when_valid_input_then_execute_successfully(pid, op_param,
                                                                                       validate_param_return,
                                                                                       expected_execute_call):
    # Arrange
    op = OpcheckAllReduceOperation()
    op.pid = pid
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test_all_reduce()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
