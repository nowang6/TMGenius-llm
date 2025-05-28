from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckLinearOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'transposeA': False, 'transposeB': True, 'hasBias': True}, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    ({'transposeA': True, 'transposeB': False, 'hasBias': True}, [torch.randn(2, 3), torch.randn(2, 3)], RuntimeError),
])
def test_golden_calc_when_invalid_input(op_param, in_tensors, expected_error):
    # Arrange
    op = OpcheckLinearOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'transposeA': False, 'transposeB': True, 'hasBias': False}, True, True),
    ({'transposeA': True, 'transposeB': False, 'hasBias': False}, True, True),
    ({'transposeA': False, 'transposeB': True, 'hasBias': True}, True, True),
    ({'transposeA': True, 'transposeB': False, 'hasBias': True}, True, True),
    ({'transposeA': False, 'transposeB': True}, False, False),
    ({'transposeA': True, 'transposeB': False}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckLinearOperation()
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
