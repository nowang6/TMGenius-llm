from unittest.mock import patch

import torch
import pytest

from msit_llm.opcheck.check_case import OpcheckAllGatherOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckAllGatherOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("in_tensors, expected_shape", [
    ([torch.randn(4, 8), torch.randn(4, 8)], (2, 4, 8)),
    ([torch.randn(4, 8), torch.randn(4, 9)], None),  # Mismatched shapes
    ([], None),  # Empty list
])
def test_golden_calc_given_in_tensors_when_valid_input_then_correct_shape(in_tensors, expected_shape):
    # Arrange
    op = OpcheckAllGatherOperation()

    def get_new_in_tensors():
        return in_tensors

    op.get_new_in_tensors = get_new_in_tensors

    # Act
    if expected_shape is None:
        with pytest.raises(RuntimeError):
            op.golden_calc(None)
    else:
        result = op.golden_calc(None)
        # Assert
        assert result[0].shape == expected_shape


@pytest.mark.parametrize("pid, op_param, validate_param_return, expected_execute_call", [
    (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, True),
    (None, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, True, False),
    (1, {'rank': 0, 'rankRoot': 0, 'rankSize': 2}, False, False),
    (1, {'rankRoot': 0, 'rankSize': 2}, False, False),
    (1, {'rank': 0, 'rankSize': 2}, False, False),
    (1, {'rank': 0, 'rankRoot': 0}, False, False),
])
def test_test_all_gather_given_pid_op_param_when_valid_input_then_execute_successfully(pid, op_param,
                                                                                       validate_param_return,
                                                                                       expected_execute_call):
    # Arrange
    op = OpcheckAllGatherOperation()
    op.pid = pid
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test_all_gather()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
