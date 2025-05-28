from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.activation import OpcheckActivationOperation, ActivationType, GeLUMode
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckActivationOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("activation_type, op_param, in_tensors, expected_result", [
    (ActivationType.ACTIVATION_RELU.value, {}, [torch.tensor([-1.0, 2.0, 0.0])], [torch.tensor([0.0, 2.0, 0.0])]),
])
def test_golden_calc_given_activation_type_op_param_in_tensors_when_valid_input_then_correct_result(activation_type,
                                                                                                    op_param,
                                                                                                    in_tensors,
                                                                                                    expected_result):
    # Arrange
    op = OpcheckActivationOperation()
    op.op_param = {'activationType': activation_type, **op_param}

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.allclose(result[0], expected_result[0], atol=1e-4)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'activationType': ActivationType.ACTIVATION_RELU.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_GELU.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_FAST_GELU.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_SWISH.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_LOG.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_SWIGLU_FORWARD.value}, True, True),
    ({'activationType': ActivationType.ACTIVATION_SWIGLU_BACKWARD.value}, True, True),
    ({}, False, False),
    ({'activationType': ActivationType.ACTIVATION_UNDEFINED.value}, False, False),
    ({'activationType': ActivationType.ACTIVATION_MAX.value}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckActivationOperation()
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
