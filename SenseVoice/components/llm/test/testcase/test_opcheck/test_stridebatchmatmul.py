from unittest.mock import patch, MagicMock

import pytest
import torch

from msit_llm.opcheck.check_case.stridebatchmatmul import OpcheckStridedBatchMatmulOperation

# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def execute(self):
        pass

OpcheckStridedBatchMatmulOperation.__bases__ = (MockOperationTest,)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown

def test_test_add_bmm1_when_execute_called_then_no_exception():
    op = OpcheckStridedBatchMatmulOperation()
    op.execute = MagicMock()

    try:
        op.test_add_bmm1()
    except Exception as e:
        pytest.fail(f"test_add_bmm1 raised an exception: {e}")

def test_golden_calc_given_tensors_when_zero_dimensions_then_empty_result():
    op = OpcheckStridedBatchMatmulOperation()
    batch, head_num = 0, 0
    m, n, k = [], [], []
    lda, ldb, ldc = [], [], []
    stridea, strideb, stridec = [], [], []
    trans_a, trans_b = False, False
    
    a = torch.tensor([], dtype=torch.float16)
    b = torch.tensor([], dtype=torch.float16)
    
    op.op_param = {
        "batch": batch,
        "head_num": head_num,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "m": m,
        "n": n,
        "k": k,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "strideA": stridea,
        "strideB": strideb,
        "strideC": stridec
    }

    result = op.golden_calc([a, b])
    
    assert len(result[0]) == 0

def test_golden_calc_given_tensors_when_invalid_batch_size_then_raise_index_error():
    op = OpcheckStridedBatchMatmulOperation()
    batch, head_num = 1, 1
    m, n, k = [3], [4], [5]
    lda, ldb, ldc = [5], [4], [3]
    stridea, strideb, stridec = [0], [0], [0]
    trans_a, trans_b = False, False
    
    a = torch.randn(15, dtype=torch.float16)
    b = torch.randn(20, dtype=torch.float16)
    
    op.op_param = {
        "batch": batch + 1,  # Invalid batch size
        "head_num": head_num,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "m": m,
        "n": n,
        "k": k,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "strideA": stridea,
        "strideB": strideb,
        "strideC": stridec
    }

    with pytest.raises(IndexError):
        op.golden_calc([a, b])

def test_golden_calc_given_tensors_when_mismatched_dimensions_then_raise_runtime_error():
    op = OpcheckStridedBatchMatmulOperation()
    batch, head_num = 1, 1
    m, n, k = [3], [4], [5]
    lda, ldb, ldc = [6], [4], [3]  # lda does not match m or k
    stridea, strideb, stridec = [0], [0], [0]
    trans_a, trans_b = False, False
    
    a = torch.randn(15, dtype=torch.float16)
    b = torch.randn(20, dtype=torch.float16)
    
    op.op_param = {
        "batch": batch,
        "head_num": head_num,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "m": m,
        "n": n,
        "k": k,
        "lda": lda,
        "ldb": ldb,
        "ldc": ldc,
        "strideA": stridea,
        "strideB": strideb,
        "strideC": stridec
    }

    with pytest.raises(RuntimeError):
        op.golden_calc([a, b])

def test_golden_calc_given_invalid_input_when_missing_parameters_then_raise_exception():
    op = OpcheckStridedBatchMatmulOperation()
    op.op_param = {}
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float16)
    b = torch.tensor([1, 2, 3, 4], dtype=torch.float16)
    in_tensors = [a, b]
    
    with pytest.raises(Exception):
        op.golden_calc(in_tensors)
