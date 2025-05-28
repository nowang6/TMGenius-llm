from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.linear_parallel import OpcheckLinearParallelOperation, ParallelType, QuantType
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearParallelOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, expected_execute_call", [
    ({'backend': 'hccl', 'rank': 0, 'rankSize': 2}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_ALL_REDUCE.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.LINEAR_REDUCE_SCATTER.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.ALL_GATHER_LINEAR.value}, True),
    ({'backend': 'lcoc', 'rank': 0, 'rankSize': 2, 'type': ParallelType.PURE_LINEAR.value}, True),
])
def test_test_when_valid_input(op_param, expected_execute_call):
    op = OpcheckLinearParallelOperation()
    op.op_param = op_param
    with patch.object(op, 'get_soc_version', return_value='Ascend910B'):
        with patch.object(op, 'validate_param', return_value=True):
            with patch.object(op, 'execute') as mock_execute:
                op.test()
                if expected_execute_call:
                    mock_execute.assert_called_once()
                else:
                    mock_execute.assert_not_called()
