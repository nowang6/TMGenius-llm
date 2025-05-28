from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckConcatOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckConcatOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_shape", [
    ({'concatDim': 0}, [torch.randn(2, 3), torch.randn(3, 3)], (5, 3)),
    ({'concatDim': 1}, [torch.randn(2, 3), torch.randn(2, 4)], (2, 7)),
    ({'concatDim': -1}, [torch.randn(2, 3), torch.randn(2, 4)], (2, 7)),
    ({'concatDim': -2}, [torch.randn(2, 3), torch.randn(3, 3)], (5, 3)),
])
def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_shape(op_param, in_tensors,
                                                                                   expected_shape):
    # Arrange
    op = OpcheckConcatOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == expected_shape


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'concatDim': 0}, [torch.randn(2, 3), torch.randn(2, 4)], RuntimeError),
    ({'concatDim': 1}, [torch.randn(2, 3), torch.randn(3, 3)], RuntimeError),
])
def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(op_param, in_tensors,
                                                                                   expected_error):
    # Arrange
    op = OpcheckConcatOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'concatDim': 0}, True, True),
    ({'concatDim': 1}, True, True),
    ({'concatDim': -1}, True, True),
    ({'concatDim': -2}, True, True),
    ({}, False, False),
    ({'concatDim': 0}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckConcatOperation()
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
