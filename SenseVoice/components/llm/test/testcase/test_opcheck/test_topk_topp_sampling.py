from unittest.mock import patch, MagicMock

import pytest
import torch

from msit_llm.opcheck.check_case.topk_topp_sampling import OpcheckToppOperation, TopkToppSamplingType


# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def execute(self):
        pass

OpcheckToppOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_single_topk_sampling_given_invalid_input_when_empty_probs_then_raise_exception():
    op = OpcheckToppOperation()
    op.op_param = {'topk': 5, 'randSeed': 0}
    in_tensors = [torch.tensor([]), torch.tensor([0.5])]
    
    with pytest.raises(Exception):
        op.single_topk_sampling(in_tensors)

def test_sampling_undefined_given_invalid_input_when_empty_probs_then_raise_exception():
    op = OpcheckToppOperation()
    op.op_param = {}
    in_tensors = [torch.tensor([]), torch.tensor([2]), torch.tensor([0.5]), torch.tensor([1.0])]
    
    with pytest.raises(Exception):
        op.sampling_undefined(in_tensors, TopkToppSamplingType.BATCH_TOPK_EXPONENTIAL_SAMPLING.value)

def test_golden_calc_given_invalid_input_when_empty_probs_then_raise_exception():
    op = OpcheckToppOperation()
    op.op_param = {'topktopp_sampling_type': TopkToppSamplingType.SINGLE_TOPK_SAMPLING.value, 'topk': 5, 'randSeed': 0}
    in_tensors = [torch.tensor([]), torch.tensor([0.5])]
    
    with pytest.raises(Exception):
        op.golden_calc(in_tensors)

def test_golden_calc_given_probs_when_sampling_type_single_topk_then_calls_single_topk_sampling():
    op = OpcheckToppOperation()
    op.op_param = {'topktopp_sampling_type': TopkToppSamplingType.SINGLE_TOPK_SAMPLING.value}

    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    topp = torch.tensor([0.5, 0.6])
    in_tensors = [probs, topp]

    with patch.object(op, 'single_topk_sampling', return_value=[torch.zeros((2, 1), dtype=torch.int32), 
                                                                torch.zeros((2, 1), dtype=torch.float16)]) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors)

def test_golden_calc_given_probs_when_sampling_type_undefined_then_calls_sampling_undefined():
    op = OpcheckToppOperation()
    op.op_param = {'topktopp_sampling_type': TopkToppSamplingType.SAMPLING_UNDEFINED.value}

    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    topk = torch.tensor([[3], [3]])
    topp = torch.tensor([0.5, 0.6])
    in_tensors = [probs, topk, topp]

    with patch.object(op, 'sampling_undefined', return_value=[torch.zeros((2, 1), dtype=torch.int32), 
                                                              torch.zeros((2, 1), dtype=torch.float16)]) as mock_method:
        result = op.golden_calc(in_tensors)

    mock_method.assert_called_once_with(in_tensors, TopkToppSamplingType.SAMPLING_UNDEFINED.value)

def test_test_when_validate_param_fails_then_not_execute():
    op = OpcheckToppOperation()
    op.validate_param = MagicMock(return_value=False)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_not_called()

def test_test_when_validate_param_passes_then_execute_called():
    op = OpcheckToppOperation()
    op.validate_param = MagicMock(return_value=True)
    op.execute = MagicMock()

    op.test()

    op.execute.assert_called_once()