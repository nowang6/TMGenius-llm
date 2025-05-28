import pytest
import torch
from msit_llm.opcheck.check_case.pad import OpcheckPadOperation

from mock_operation_test import MockOperationTest

OpcheckPadOperation.__bases__ = (MockOperationTest,)

def test_golden_calc_given_valid_input_when_batch_1_then_correct_output():
    # Arrange
    op = OpcheckPadOperation()
    in_tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # tmp_out
        torch.tensor([0]),  # padding_offset
        torch.tensor([[2]]),  # seq_len
        torch.tensor([[1, 2]])  # input_ids
    ]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    expected = torch.tensor([[3.0, 4.0]])
    assert torch.equal(result[0], expected)

def test_golden_calc_given_invalid_input_when_seq_len_mismatch_then_raise_error():
    # Arrange
    op = OpcheckPadOperation()
    in_tensors = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # tmp_out
        torch.tensor([0]),  # padding_offset
        torch.tensor([[3]]),  # seq_len
        torch.tensor([[1, 2]])  # input_ids
    ]

    # Act & Assert
    with pytest.raises(IndexError):
        op.golden_calc(in_tensors)
