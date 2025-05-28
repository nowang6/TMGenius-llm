from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.layer_norm import OpcheckLayerNormOperation, LayerNormType, QuantType, DynamicQuantType
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckLayerNormOperation.__bases__ = (MockOperationTest,)


# Mock the logger to capture log messages
@pytest.fixture
def mock_logger():
    with patch.object(logger, 'info') as mock_info, patch.object(logger, 'debug') as mock_debug:
        yield mock_info, mock_debug


@pytest.mark.parametrize("layer_type, validate_param_return, expected_execute_call", [
    (LayerNormType.LAYER_NORM_NROM.value, True, True),
    (LayerNormType.LAYER_NORM_PRENORM.value, True, True),
    (LayerNormType.LAYER_NORM_POSTNORM.value, True, True),
    (LayerNormType.LAYER_NORM_UNDEFINED.value, False, False),
])
def test_test_when_valid_input(layer_type, validate_param_return, expected_execute_call):
    op = OpcheckLayerNormOperation()
    op.op_param = {'layerType': layer_type}

    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
