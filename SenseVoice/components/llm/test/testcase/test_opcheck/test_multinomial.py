import pytest
import torch
import torch_npu
from msit_llm.opcheck.check_case.multinomial import OpcheckMultinomialOperation

from mock_operation_test import MockOperationTest


OpcheckMultinomialOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_golden_calc_given_valid_input_when_numSamples_1_then_correct_shape():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (4, 1)

def test_golden_calc_given_valid_input_when_numSamples_5_then_correct_shape():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 5, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (4, 5)

def test_golden_calc_given_invalid_input_when_numSamples_negative_then_raise_exception():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": -1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    # Act & Assert
    with pytest.raises(Exception):
        op.golden_calc(in_tensors)

def test_golden_calc_given_valid_input_when_randSeed_non_zero_then_correct_shape():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 12345}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (4, 1)

def test_golden_calc_given_valid_input_when_randSeed_zero_then_correct_shape():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1, "randSeed": 0}
    input0 = torch.randn(4, 16)
    in_tensors = [input0]

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    assert result[0].shape == (4, 1)


def test_test_given_missing_numSamples_when_randSeed_provided_then_return_early():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"randSeed": 0}

    # Act
    op.test()

    # Assert (No exception raised)
    assert True

def test_test_given_missing_randSeed_when_numSamples_provided_then_return_early():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {"numSamples": 1}

    # Act
    op.test()

    # Assert (No exception raised)
    assert True

def test_test_given_missing_both_params_when_no_params_provided_then_return_early():
    # Arrange
    op = OpcheckMultinomialOperation()
    op.op_param = {}

    # Act
    op.test()

    # Assert (No exception raised)
    assert True