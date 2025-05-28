from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckIndexAddOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckIndexAddOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_result", [
    ({'indexType': 1, 'axis': 0}, [torch.randn(2, 4, 3, 3), torch.tensor([0,1]), torch.randn(2, 4, 3, 3), torch.tensor(1.0)], True),
    ({'indexType': 0, 'axis': 0}, [torch.randn(2, 4, 3, 3), torch.tensor([0,1]), torch.randn(2, 4, 3, 3), torch.tensor(1.0)], False),
])
def test_golden_calc_when_valid_input(op_param, in_tensors, expected_result):
    # Arrange
    op = OpcheckIndexAddOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    if expected_result:
        assert result
        assert len(result) > 0
    else:
        assert not result


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'indexType': 1, 'axis': 0}, True, True),
    ({'indexType': 1, 'axis': 1}, True, True),
    ({'indexType': 0, 'axis': 0}, True, True),
    ({'indexType': 0, 'axis': 1}, True, True),
    ({'indexType': 1}, False, False),
    ({'axis': 0}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckIndexAddOperation()
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
