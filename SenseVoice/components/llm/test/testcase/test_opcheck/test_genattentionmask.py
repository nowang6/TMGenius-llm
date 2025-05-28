from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckElewiseSubOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckElewiseSubOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'seqLen': [2, 3], 'headNum': 2}, True, True),
    ({'seqLen': [1, 2], 'headNum': 1}, True, True),
    ({'seqLen': [3, 3], 'headNum': 3}, True, True),
    ({'seqLen': [2, 3]}, False, False),
    ({'headNum': 2}, False, False),
    ({}, False, False),
])
def test_test_2d_half_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckElewiseSubOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test_2d_half()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
