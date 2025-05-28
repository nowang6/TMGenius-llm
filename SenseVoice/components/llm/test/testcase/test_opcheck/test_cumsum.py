from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckCumsumOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckCumsumOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_result", [
    ({'axes': [0]}, [torch.tensor([1.0, 2.0, 3.0])], [torch.tensor([1.0, 3.0, 6.0])]),
    ({'axes': [1]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], [torch.tensor([[1.0, 3.0], [3.0, 7.0]])]),
    ({'axes': [0]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], [torch.tensor([[1.0, 2.0], [4.0, 6.0]])]),
])
def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_result(op_param, in_tensors,
                                                                                    expected_result):
    # Arrange
    op = OpcheckCumsumOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.equal(result[0], expected_result[0])


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'axes': [2]}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], IndexError),
])
def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(op_param, in_tensors,
                                                                                   expected_error):
    # Arrange
    op = OpcheckCumsumOperation()
    op.op_param = op_param

    # Act & Assert
    if expected_error:
        with pytest.raises(expected_error):
            op.golden_calc(in_tensors)
    else:
        result = op.golden_calc(in_tensors)
        assert torch.equal(result[0], in_tensors[0])


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'axes': [0]}, True, True),
    ({'axes': [1]}, True, True),
    ({'axes': [0]}, False, False),
    ({'axes': [1]}, False, False),
    ({}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckCumsumOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
