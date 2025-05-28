import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.nonzero import OpcheckNonzeroOperation
from mock_operation_test import MockOperationTest


OpcheckNonzeroOperation.__bases__ = (MockOperationTest,)

def test_golden_calc_given_positive_tensor_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[1, 2], [3, 4]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()

def test_golden_calc_given_mixed_tensor_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[1, -2], [0, 4]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(3).long()

def test_golden_calc_given_all_zero_tensor_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[0, 0], [0, 0]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(0).long()

def test_golden_calc_given_empty_tensor_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 0)
    assert result[1] == torch.tensor(0).long()

def test_golden_calc_given_tensor_with_nan_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[1, float('nan')], [float('nan'), 4]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()

def test_golden_calc_given_tensor_with_inf_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[1, float('inf')], [float('inf'), 4]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()

def test_golden_calc_given_tensor_with_negative_inf_when_valid_input_then_correct_result():
    # Arrange
    in_tensors = [torch.tensor([[1, float('-inf')], [float('-inf'), 4]])]
    op = OpcheckNonzeroOperation()

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (2, 4)
    assert result[1] == torch.tensor(4).long()
