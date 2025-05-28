import pytest
import torch
from msit_llm.opcheck.check_case.repeat import OpcheckRepeatOperation

from mock_operation_test import MockOperationTest


OpcheckRepeatOperation.__bases__ = (MockOperationTest,)


def test_golden_calc_given_valid_multiples_when_1d_tensor_then_correct_result():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2,)}
    in_tensors = [torch.tensor([1, 2, 3])]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    expected = torch.tensor([1, 2, 3, 1, 2, 3])
    assert torch.allclose(result[0], expected)

def test_golden_calc_given_valid_multiples_when_2d_tensor_then_correct_result():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 3)}
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    expected = torch.tensor([[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])
    assert torch.allclose(result[0], expected)

def test_golden_calc_given_valid_multiples_when_3d_tensor_then_correct_result():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 2, 2)}
    in_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    expected = torch.tensor([[[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
                             [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]],
                             [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]],
                             [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]]])
    assert torch.allclose(result[0], expected)


def test_golden_calc_given_invalid_multiples_when_3d_tensor_then_error():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2, 2)}
    in_tensors = [torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])]

    # Act & Assert
    with pytest.raises(RuntimeError):
        op.golden_calc(in_tensors)

def test_test_given_valid_multiples_when_execute_then_no_error():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": (2,)}

    def mock_validate_param(*args, **kwargs):
        True

    op.validate_param = mock_validate_param

    def mock_execute():
        pass

    op.execute = mock_execute

    # Act
    op.test()

    # Assert
    assert True

def test_test_given_invalid_multiples_when_execute_then_return():
    # Arrange
    op = OpcheckRepeatOperation()
    op.op_param = {"multiples": None}

    def mock_validate_param(*args, **kwargs):
        False

    op.validate_param = mock_validate_param

    # Act
    result = op.test()

    # Assert
    assert result is None