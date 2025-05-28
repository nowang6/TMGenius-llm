from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckFillOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckFillOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_result", [
    ({'withMask': True, 'value': [1.0]}, [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False, True])],
     [torch.tensor([1.0, 2.0, 1.0])]),
    ({'withMask': True, 'value': [0.0]}, [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([False, True, False])],
     [torch.tensor([1.0, 0.0, 3.0])]),
])
def test_golden_calc_given_op_param_in_tensors_when_valid_input_then_correct_result(op_param, in_tensors,
                                                                                    expected_result):
    # Arrange
    op = OpcheckFillOperation()
    op.op_param = op_param

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert torch.allclose(result[0], expected_result[0], atol=1e-4)


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'withMask': True, 'value': [1.0]}, [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([True, False])], RuntimeError),
])
def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(op_param, in_tensors,
                                                                                   expected_error):
    # Arrange
    op = OpcheckFillOperation()
    op.op_param = op_param

    # Act & Assert
    with pytest.raises(expected_error):
        op.golden_calc(in_tensors)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'withMask': True, 'value': [1.0]}, True, True),
    ({'withMask': False, 'outDim': [2, 2], 'value': [0.0]}, True, True),
    ({'withMask': True, 'value': [0.0]}, True, True),
    ({'withMask': False, 'outDim': [3, 3], 'value': [1.0]}, True, True),
    ({'withMask': True}, False, False),
    ({'withMask': False, 'outDim': [2, 2]}, False, False),
    ({'withMask': True, 'value': [1.0]}, False, False),
    ({'withMask': False, 'outDim': [2, 2], 'value': []}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckFillOperation()
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
