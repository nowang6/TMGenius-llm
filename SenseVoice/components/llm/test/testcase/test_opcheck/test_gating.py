from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckGatingOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckGatingOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'topkExpertNum': 2, 'cumSumNum': 3}, True, True),
    ({'topkExpertNum': 1, 'cumSumNum': 2}, True, True),
    ({'topkExpertNum': 3, 'cumSumNum': 4}, True, True),
    ({'topkExpertNum': 2}, False, False),
    ({'cumSumNum': 3}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckGatingOperation()
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
