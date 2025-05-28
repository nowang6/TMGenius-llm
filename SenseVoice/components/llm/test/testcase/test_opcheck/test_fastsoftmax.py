from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case import OpcheckFastSoftMaxOperation
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckFastSoftMaxOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("op_param, in_tensors, expected_error", [
    ({'qSeqLen': [2, 2], 'headNum': 1}, [torch.tensor([[1.0, 2.0]])], RuntimeError),
    ({'qSeqLen': [3, 3], 'headNum': 1}, [torch.tensor([[1.0, 2.0, 3.0]])], RuntimeError),
    ({'qSeqLen': [2, 2], 'headNum': 2}, [torch.tensor([[1.0, 2.0], [3.0, 4.0]])], RuntimeError),
])
def test_golden_calc_given_op_param_in_tensors_when_invalid_input_then_raise_error(op_param, in_tensors,
                                                                                   expected_error):
    # Arrange
    op = OpcheckFastSoftMaxOperation()
    op.op_param = op_param

    # Act & Assert
    if expected_error:
        with pytest.raises(expected_error):
            op.golden_calc(in_tensors)
    else:
        result = op.golden_calc(in_tensors)
        assert torch.allclose(result[0], in_tensors[0], atol=1e-4)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'qSeqLen': [2, 2], 'headNum': 1}, True, True),
    ({'qSeqLen': [3, 3], 'headNum': 1}, True, True),
    ({'qSeqLen': [2, 2], 'headNum': 2}, True, True),
    ({'qSeqLen': [2, 2]}, False, False),
    ({'headNum': 1}, False, False),
    ({}, False, False),
])
def test_test_fastsoftmax_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                                    expected_execute_call):
    # Arrange
    op = OpcheckFastSoftMaxOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test_fastsoftmax()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
