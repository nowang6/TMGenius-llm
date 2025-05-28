from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckGatherOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckGatherOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'axis': 0, 'batchDims': 0}, True, True),
    ({'axis': 1, 'batchDims': 0}, True, True),
    ({'axis': 0, 'batchDims': 1}, True, True),
    ({'axis': 1, 'batchDims': 1}, True, True),
    ({'axis': 0}, False, False),
    ({'batchDims': 0}, False, False),
    ({}, False, False),
])
def test_test_when_valid_input(op_param, validate_param_return, expected_execute_call):
    # Arrange
    op = OpcheckGatherOperation()
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
