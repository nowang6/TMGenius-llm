from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckKvCacheOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckKvCacheOperation.__bases__ = (MockOperationTest,)


def test_test_when_valid_input():
    # Arrange
    op = OpcheckKvCacheOperation()
    op.op_param = {}

    # Act
    with patch.object(op, 'execute_inplace') as mock_execute:
        op.test()

    # Assert
    mock_execute.assert_called_once()
