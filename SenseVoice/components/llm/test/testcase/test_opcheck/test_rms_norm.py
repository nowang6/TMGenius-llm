from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.rms_norm import OpcheckRmsNormOperation, RmsNormType, QuantType, DynamicQuantType
from mock_operation_test import MockOperationTest

OpcheckRmsNormOperation.__bases__ = (MockOperationTest,)


def test_validate_rmsnorm_param_given_valid_layer_type_when_validate_then_correct_result():
    # Arrange
    op = OpcheckRmsNormOperation()
    op.op_param = {'layerType': RmsNormType.RMS_NORM_NORM.value, 'normParam': {'quantType': QuantType.QUANT_INT8.value}}

    # Act
    result = op.validate_rmsnorm_param(RmsNormType.RMS_NORM_NORM.value)

    # Assert
    assert result['quantType'] == QuantType.QUANT_INT8.value


@pytest.mark.parametrize("op_param, expected_execute_call", [
    ({'layerType': RmsNormType.RMS_NORM_NORM.value,
      'normParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
    ({'layerType': RmsNormType.RMS_NORM_PRE_NORM.value,
      'preNormParam': {'quantType': QuantType.QUANT_INT8.value, 'epsilon': 1e-5}}, True),
    ({'layerType': RmsNormType.RMS_NORM_POST_NORM.value,
      'postNormParam': {'quantType': QuantType.QUANT_UNDEFINED.value, 'epsilon': 1e-5}}, True),
])
def test_test_when_valid_input(op_param, expected_execute_call):
    op = OpcheckRmsNormOperation()
    op.op_param = op_param
    with patch.object(op, 'validate_param', return_value=True):
        with patch.object(op, 'execute') as mock_execute:
            op.test()
            if expected_execute_call:
                mock_execute.assert_called_once()
            else:
                mock_execute.assert_not_called()
