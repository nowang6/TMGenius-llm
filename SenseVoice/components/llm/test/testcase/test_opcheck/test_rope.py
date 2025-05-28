import pytest
import torch
from msit_llm.opcheck.check_case.rope import OpcheckUnpadRopeOperation

from mock_operation_test import MockOperationTest


OpcheckUnpadRopeOperation.__bases__ = (MockOperationTest,)

def test_rotate_half_given_valid_input_when_even_dim_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Act
    result = op.rotate_half(x)

    # Assert
    expected = torch.tensor([[-2.0, 1.0], [-4.0, 3.0]])
    assert torch.equal(result, expected)


def test_golden_func1_given_valid_input_when_batch_3_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4, 4, 4])  # seqlen
    ]

    # Act
    result = op.golden_func1(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_func2_given_valid_input_when_batch_3_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    # Act
    result = op.golden_func2(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_func3_given_valid_input_when_batch_3_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    # Act
    result = op.golden_func3(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_func4_given_valid_input_when_batch_3_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    # Act
    result = op.golden_func4(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_calc_given_valid_input_when_rotaryCoeff_4_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 4}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4, 4, 4])  # seqlen
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_calc_given_valid_input_when_rotaryCoeff_64_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 64}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)

def test_golden_calc_given_valid_input_when_rotaryCoeff_none_then_correct_output():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}
    in_tensors = [
        torch.randn(12, 8),  # q
        torch.randn(12, 8),  # k
        torch.randn(12, 4),  # cos
        torch.randn(12, 4),  # sin
        torch.tensor([4])  # seqlen
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert len(result) == 2
    assert result[0].shape == (12, 8)
    assert result[1].shape == (12, 8)


def test_test_given_invalid_params_when_missing_rotaryCoeff_then_return_early():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}

    # Act
    result = op.test()

    # Assert
    assert result is None

def test_validate_param_given_valid_params_when_rotaryCoeff_4_then_return_true():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {'rotaryCoeff': 4}

    # Act
    result = op.validate_param("rotaryCoeff")

    # Assert
    assert result is True

def test_validate_param_given_invalid_params_when_missing_rotaryCoeff_then_return_false():
    # Arrange
    op = OpcheckUnpadRopeOperation()
    op.op_param = {}

    # Act
    result = op.validate_param("rotaryCoeff")

    # Assert
    assert result is False