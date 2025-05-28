from unittest.mock import patch
from dataclasses import dataclass

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckLinearSparseOperation
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLinearSparseOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


@dataclass
class InvalidTestParams:
    transposeA: bool
    transposeB: bool
    in_tensors: list
    expected_error: type


@pytest.mark.parametrize("params", [
    InvalidTestParams(False, True, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(True, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
    InvalidTestParams(False, False, [torch.randn(2, 3), torch.randn(3, 2)], RuntimeError),
])
def test_golden_calc_when_invalid_input(mock_logger, params):
    op = OpcheckLinearSparseOperation()
    op.op_param = {"transposeA": params.transposeA, "transposeB": params.transposeB}
    with pytest.raises(params.expected_error):
        op.golden_calc(params.in_tensors)
