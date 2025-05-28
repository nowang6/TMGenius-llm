from unittest.mock import patch, MagicMock

import pytest
import torch

from msit_llm.opcheck.check_case.transdata import OpcheckTransdataOperation, TransdataType


# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def execute(self):
        pass

OpcheckTransdataOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_round_up_given_x_align_when_valid_input_then_correct_output():
    result = OpcheckTransdataOperation.round_up(10, 4)
    assert result == 12

def test_round_up_given_x_align_when_align_zero_then_minus_one():
    result = OpcheckTransdataOperation.round_up(10, 0)
    assert result == -1

def test_custom_pad_given_tensor_pad_dims_when_valid_input_then_padded_tensor():
    tensor = torch.randn(2, 3)
    pad_dims = [0, 0, 1, 1]
    padded_tensor = OpcheckTransdataOperation.custom_pad(tensor, pad_dims)
    assert padded_tensor.shape == (4, 3)

def test_custom_reshape_given_tensor_target_shape_when_valid_input_then_reshaped_tensor():
    tensor = torch.randn(2, 6)
    target_shape = (3, 4)
    reshaped_tensor = OpcheckTransdataOperation.custom_reshape(tensor, target_shape)
    assert reshaped_tensor.shape == target_shape

def test_custom_transpose_given_tensor_dim1_dim2_when_valid_input_then_transposed_tensor():
    tensor = torch.randn(2, 3)
    transposed_tensor = OpcheckTransdataOperation.custom_transpose(tensor, 0, 1)
    assert transposed_tensor.shape == (3, 2)

def test_golden_nd_to_nz_3d_given_tensors_when_valid_input_then_correct_output():
    tensor = torch.randn(2, 3, 4)
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nd_to_nz_3d(in_tensors)

    assert result.shape == (2, 1, 16, 16)

def test_golden_nd_to_nz_2d_given_tensors_when_valid_input_then_correct_output():
    tensor = torch.randn(2, 3)
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nd_to_nz_2d(in_tensors)

    assert result.shape == (1, 1, 16, 16)

def test_golden_nz_to_nd_given_tensors_out_crops_when_valid_input_then_correct_output():
    tensor = torch.randn(2, 3, 4, 8)
    out_crops = [3, 4]
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nz_to_nd(in_tensors, out_crops)

    assert result.shape == (2, 3, 4)

def test_golden_calc_given_tensors_when_nd_to_nz_3d_then_calls_golden_nd_to_nz_3d():
    op = OpcheckTransdataOperation()
    op.op_param = {"transdataType": TransdataType.ND_TO_FRACTAL_NZ.value}
    tensor = torch.randn(2, 3, 4)
    in_tensors = [tensor]

    with patch.object(OpcheckTransdataOperation, 'golden_nd_to_nz_3d', return_value=tensor) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors)

def test_golden_calc_given_tensors_when_nd_to_nz_2d_then_calls_golden_nd_to_nz_2d():
    op = OpcheckTransdataOperation()
    op.op_param = {"transdataType": TransdataType.ND_TO_FRACTAL_NZ.value}
    tensor = torch.randn(2, 3)
    in_tensors = [tensor]

    with patch.object(OpcheckTransdataOperation, 'golden_nd_to_nz_2d', return_value=tensor) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors)

def test_golden_calc_given_tensors_when_nz_to_nd_then_calls_golden_nz_to_nd():
    op = OpcheckTransdataOperation()
    op.op_param = {"transdataType": TransdataType.FRACTAL_NZ_TO_ND.value, "outCrops": [3, 4]}
    tensor = torch.randn(2, 3, 4, 8)
    in_tensors = [tensor]

    with patch.object(OpcheckTransdataOperation, 'golden_nz_to_nd', return_value=tensor) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors, [3, 4])

def test_test_when_validate_param_fails_then_not_execute():
    op = OpcheckTransdataOperation()
    op.validate_param = MagicMock(return_value=False)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_not_called()

def test_test_when_validate_param_passes_then_execute_called():
    op = OpcheckTransdataOperation()
    op.validate_param = MagicMock(return_value=True)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_called_once()

def test_golden_nd_to_nz_3d_given_int8_tensor_when_valid_input_then_correct_output():
    tensor = torch.randint(-128, 127, (2, 3, 4), dtype=torch.int8)
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nd_to_nz_3d(in_tensors)
    
    expected_shape = (2, 1 ,16, 32)
    assert result.shape == expected_shape

def test_golden_nd_to_nz_2d_given_int8_tensor_when_valid_input_then_correct_output():
    tensor = torch.randint(-128, 127, (2, 3), dtype=torch.int8)
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nd_to_nz_2d(in_tensors)

    expected_shape = (1, 1, 16, 32)
    assert result.shape == expected_shape

def test_golden_nz_to_nd_given_int8_tensor_out_crops_when_valid_input_then_correct_output():
    tensor = torch.randint(-128, 127, (2, 3, 4, 8), dtype=torch.int8)
    out_crops = [3, 4]
    in_tensors = [tensor]

    result = OpcheckTransdataOperation.golden_nz_to_nd(in_tensors, out_crops)

    assert result.shape == (2, 3, 4)

def test_golden_calc_given_invalid_transdata_type_when_invalid_input_then_default_behavior():
    op = OpcheckTransdataOperation()
    op.op_param = {"transdataType": 999}  # Invalid type
    tensor = torch.randn(2, 3, 4)
    in_tensors = [tensor]

    with patch.object(OpcheckTransdataOperation, 'golden_nz_to_nd', return_value=tensor) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors, None)

    mock_method.assert_called_once_with(in_tensors, None)