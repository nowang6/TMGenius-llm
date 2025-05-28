import pytest
import torch
from msit_llm.opcheck.check_case.onehot import OpcheckOnehotOperation

from mock_operation_test import MockOperationTest


OpcheckOnehotOperation.__bases__ = (MockOperationTest,)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_golden_calc_given_valid_input_when_axis_0_depth_3_then_correct_output():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0, 'depth': 3}
    in_tensors = torch.tensor([0, 1, 2])

    # Act
    result = op.golden_calc([in_tensors])

    # Assert
    expected = torch.eye(3)[in_tensors]
    assert torch.equal(result[0], expected)

def test_golden_calc_given_invalid_axis_when_axis_out_of_range_then_raise_error():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 2, 'depth': 3}
    in_tensors = torch.tensor([0, 1, 2])

    # Act & Assert
    with pytest.raises(IndexError):
        op.golden_calc([in_tensors])


def test_test_given_invalid_params_when_missing_depth_then_return_early():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0}

    # Act
    result = op.test()

    # Assert
    assert result is None

def test_test_given_invalid_params_when_missing_axis_then_return_early():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'depth': 3}

    # Act
    result = op.test()

    # Assert
    assert result is None

def test_test_given_invalid_params_when_missing_both_axis_and_depth_then_return_early():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {}

    # Act
    result = op.test()

    # Assert
    assert result is None

def test_validate_param_given_valid_params_when_axis_0_depth_3_then_return_true():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0, 'depth': 3}

    # Act
    result = op.validate_param("axis", "depth")

    # Assert
    assert result is True

def test_validate_param_given_invalid_params_when_missing_depth_then_return_false():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'axis': 0}

    # Act
    result = op.validate_param("axis", "depth")

    # Assert
    assert result is False

def test_validate_param_given_invalid_params_when_missing_axis_then_return_false():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {'depth': 3}

    # Act
    result = op.validate_param("axis", "depth")

    # Assert
    assert result is False

def test_validate_param_given_invalid_params_when_missing_both_axis_and_depth_then_return_false():
    # Arrange
    op = OpcheckOnehotOperation()
    op.op_param = {}

    # Act
    result = op.validate_param("axis", "depth")

    # Assert
    assert result is False