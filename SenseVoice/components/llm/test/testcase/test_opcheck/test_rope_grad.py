import pytest
import torch
from mock_operation_test import MockOperationTest
from msit_llm.opcheck.check_case.rope_grad import OpcheckRopeGradOperation

OpcheckRopeGradOperation.__bases__ = (MockOperationTest,)

def test_golden_calc_given_empty_in_tensors_when_execution_then_raise_exception():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [128, 128]}
    in_tensors = []

    # Act & Assert
    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)

def test_golden_calc_given_small_qSeqLen_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [64, 64]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [64, 64]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [64, 64]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_q_grad_k_grad_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_and_small_q_grad_k_grad_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128) * 0.001,  # q_grad
        torch.randn(1, 128, 128) * 0.001,  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_bs_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_and_small_qSeqLen_list_values_and_small_bs_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_qSeqLen_list_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128),  # cos
        torch.randn(128, 128)  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)

def test_golden_calc_given_small_cos_sin_values_with_small_qSeqLen_and_small_bs_and_small_qSeqLen_list_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_and_small_cos_sin_values_and_small_qSeqLen_list_values_when_execution_then_correct_result():
    # Arrange
    op = OpcheckRopeGradOperation()
    op.op_param = {'qSeqLen': [32, 32, 32, 32]}
    in_tensors = [
        torch.randn(1, 128, 128),  # q_grad
        torch.randn(1, 128, 128),  # k_grad
        torch.randn(128, 128) * 0.001,  # cos
        torch.randn(128, 128) * 0.001  # sin
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (1, 128, 128)
    assert result[1].shape == (1, 128, 128)
