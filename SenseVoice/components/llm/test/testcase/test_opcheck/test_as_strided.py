from unittest.mock import patch

import torch
import pytest

from msit_llm.opcheck.check_case import OpcheckAsStridedOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckAsStridedOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_shape, expected_error", [
    ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, [torch.randn(4, 8, 16)], (4, 8, 16), None),
    ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, [torch.randn(4, 8, 15)], (4, 8, 16), RuntimeError),
])
def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_shape(op_param, in_tensors, expected_shape,
                                                                                   expected_error):
    # Arrange
    op = OpcheckAsStridedOperation()
    op.op_param = op_param

    # Act & Assert
    if expected_error:
        with pytest.raises(expected_error):
            op.golden_calc(in_tensors)
    else:
        result = op.golden_calc(in_tensors)
        assert result[0].shape == expected_shape


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, True, True),
    ({'size': [4, 8, 16], 'stride': [128, 16, 1], 'offset': [0]}, False, False),
    ({'stride': [128, 16, 1], 'offset': [0]}, False, False),
    ({'size': [4, 8, 16], 'offset': [0]}, False, False),
    ({'size': [4, 8, 16], 'stride': [128, 16, 1]}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckAsStridedOperation()
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
