from unittest.mock import patch
import pytest
import torch
import torch_npu

from msit_llm.opcheck.check_case.paged_attention import OpcheckPagedAttentionAttentionOperation, MaskType, QuantType

from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckPagedAttentionAttentionOperation.__bases__ = (MockOperationTest,)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    yield
    # Teardown


# Test cases for mask_nz_2_nd
def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_alibi_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_ALIBI.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (1, head_num, 16, 16 * 16)


def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_norm_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.MASK_TYPE_NORM.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)


def test_mask_nz_2_nd_given_mask_and_context_lens_when_mask_type_undefined_then_correct_shape():
    mask = torch.randn(4, 8, 16, 32)
    mask_type = MaskType.UNDEFINED.value
    context_lens = [4, 8, 12, 16]
    head_num = 4
    result_mask = OpcheckPagedAttentionAttentionOperation.mask_nz_2_nd(mask, mask_type, context_lens, head_num)
    assert result_mask.shape == (4, 16, 16 * 16)


# # Test cases for get_quant_param
def test_get_quant_param_given_quant_type_has_quant_offset_when_quant_type_undefined_then_correct_values():
    op = OpcheckPagedAttentionAttentionOperation()
    quant_type = QuantType.TYPE_QUANT_UNDEFINED.value
    has_quant_offset = False
    is_int8_flag, has_bias, fp32_input = op.get_quant_param(quant_type, has_quant_offset)
    assert is_int8_flag is False
    assert has_bias is False
    assert fp32_input == [None, None, None, None]


@patch.object(OpcheckPagedAttentionAttentionOperation, 'group_matmul')
def test_ref_masked_attention_given_masked_attention_input_alibi_bias_when_quant_type_dequant_fusion_then_correct_shape(
        mock_group_matmul):
    op = OpcheckPagedAttentionAttentionOperation()
    masked_attention_input = [torch.randn(1, 8, 16), torch.randn(16, 4, 16), torch.randn(16, 4, 16), 1.0]
    alibi_bias = torch.randn(8, 1, 16)
    mock_group_matmul.return_value = torch.randn(8, 1, 16)
    out = op.ref_masked_attention(masked_attention_input, alibi_bias)
    assert out.shape == (1, 8, 16)
