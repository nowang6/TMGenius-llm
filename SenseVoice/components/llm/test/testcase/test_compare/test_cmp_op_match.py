import pytest

import msit_llm.dump.torch_dump.topo as topo
from msit_llm.compare.cmp_op_match import OpMatchMap, MatchLocation
from msit_llm.dump.torch_dump.topo import TreeNode
from msit_llm.compare.cmp_op_match import policy_enhanced_name_match, policy_layer_type_cnt_match, \
                                            policy_name_full_match, policy_output, \
                                            policy_layer_special_match, policy_rope_operator_match, \
                                            policy_module_match


@pytest.fixture
def patch_min_layer_number():
    # 记录原始值
    original_value = topo.MIN_LAYER_NUMBER
    # 修改为测试值
    topo.MIN_LAYER_NUMBER = 0
    yield
    # 恢复原始值
    topo.MIN_LAYER_NUMBER = original_value


def test_policy_name_full_match():

    golden_root = TreeNode('root.lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'golden_operation1', show_order=1)
    golden_child2 = TreeNode('child2', 'golden_operation2', show_order=2)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)

    my_root = TreeNode('LmHead', 'LmHead')
    my_child1 = TreeNode('child1', 'my_operation1', show_order=1)
    my_child2 = TreeNode('child2', 'my_operation2', show_order=2)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)

    match_map = OpMatchMap(golden_root, my_root)
    policy_name_full_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT)
    ]
    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_output():

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'golden_operation1', show_order=1)
    golden_child2 = TreeNode('child2', 'golden_operation2', show_order=2)
    golden_grandchild1 = TreeNode('grandchild1', 'golden_operation3', show_order=3)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_child2.add_child(golden_grandchild1)

    my_root = TreeNode('LmHead', 'Lmhead')
    my_child1 = TreeNode('child1', 'my_operation1', show_order=1)
    my_child2 = TreeNode('child2', 'my_operation2', show_order=2)
    my_grandchild1 = TreeNode('grandchild1', 'my_operation3', show_order=3)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_child2.add_child(my_grandchild1)

    match_map = OpMatchMap(golden_root, my_root)
    policy_output(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2.children[0], MatchLocation.ALL_OUTPUT),
        (my_child2.children[0], MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child2.children[0], MatchLocation.ALL_OUTPUT, golden_child2.children[0], MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_enhanced_name_match(patch_min_layer_number):

    assert topo.MIN_LAYER_NUMBER == 0

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_mlp = TreeNode('golden_mlp', 'mlp', tensor_path='my_golden_path')
    golden_rmsnorm = TreeNode('golden_rmsnorm', 'rmsnormoperation', 
                              tensor_path='torch_dump/root.mlp.post_attention_layernorm')
    golden_split = TreeNode('my_act', 'activationoperation', tensor_path='torch_dump/root.mlp.act_fn')
    golden_mlp.add_child(golden_rmsnorm)
    golden_mlp.add_child(golden_split)
    golden_root.add_child(golden_mlp)

    my_root = TreeNode('LmHead', 'LmHead')
    my_mlp = TreeNode('my_mlp', 'mlp', tensor_path='my_mlp_path')
    my_rmsnorm = TreeNode('my_rmsnorm', 'rmsnormoperation', 
                          tensor_path='2_MlpGateUpWeightPack/0_NormLinear/0_RmsNormOperation')
    my_split = TreeNode('my_act', 'activationoperation', tensor_path='2_MlpGateUpWeightPack/2_ActivationOperation')
    my_mlp.add_child(my_rmsnorm)
    my_mlp.add_child(my_split) 
    my_root.add_child(my_mlp)

    match_map = OpMatchMap(golden_root, my_root)
    policy_enhanced_name_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_mlp.children[0], MatchLocation.ALL_OUTPUT, golden_mlp.children[0], MatchLocation.ALL_OUTPUT),
        (my_mlp.children[1], MatchLocation.ALL_OUTPUT, golden_mlp.children[1], MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_layer_type_cnt_match():

    golden_root = TreeNode('lm_head', 'root.lm_head')
    golden_child1 = TreeNode('child1', 'LinearOperation', show_order=1)
    golden_child2 = TreeNode('child2', 'LinearNoQuant', show_order=2)
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)

    my_root = TreeNode('LmHead', 'LmHead')
    my_child1 = TreeNode('child1', 'LinearQuantOperation', show_order=1)
    my_child2 = TreeNode('child2', 'LinearDequantOnly', show_order=2)
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)

    match_map = OpMatchMap(golden_root, my_root)
    policy_layer_type_cnt_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_root, MatchLocation.ALL_OUTPUT, golden_root, MatchLocation.ALL_OUTPUT),
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected




def test_policy_layer_special_match():

    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('embed_tokens', 'golden_operation1')
    golden_child2 = TreeNode('root.model.norm', 'golden_operation2')
    golden_child3 = TreeNode('lm_head', 'golden_operation3')
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_root.add_child(golden_child3)
    
    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('wordembedding', 'my_operation1')
    my_child2 = TreeNode('RmsNormOperation_123', 'my_operation2')
    my_child3 = TreeNode('lmhead', 'my_operation3')
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_root.add_child(my_child3)

    match_map = OpMatchMap(golden_root, my_root)
    policy_layer_special_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_child1, MatchLocation.ALL_OUTPUT, golden_child1, MatchLocation.ALL_OUTPUT),
        (my_child2, MatchLocation.ALL_OUTPUT, golden_child2, MatchLocation.ALL_OUTPUT),
        (my_child3, MatchLocation.ALL_OUTPUT, golden_child3, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected


def test_policy_rope_operator_match(patch_min_layer_number):

    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('rotary1', 'rotary')
    golden_child2 = TreeNode('rotary2', 'rotary')
    golden_grandchild = TreeNode('rotary3', 'rotary')
    golden_root.add_child(golden_child1)
    golden_root.add_child(golden_child2)
    golden_child2.add_child(golden_grandchild)
    
    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('ropeoperation1', '1_RotaryPositionEmbedding')
    my_child2 = TreeNode('ropeoperation2', '1_RotaryPositionEmbedding')
    my_grandchild = TreeNode('ropeoperation3', '1_RotaryPositionEmbedding')
    my_root.add_child(my_child1)
    my_root.add_child(my_child2)
    my_child2.add_child(my_grandchild)

    match_map = OpMatchMap(golden_root, my_root)
    policy_rope_operator_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)

    expected_matches = [
        (my_grandchild, MatchLocation.ALL_INPUT, golden_grandchild, MatchLocation.ALL_OUTPUT)
    ]

    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected  


def test_policy_module_match(patch_min_layer_number):
    assert topo.MIN_LAYER_NUMBER == 0
    # 创建测试用的树
    golden_root = TreeNode('model', 'Baichuan')
    golden_child1 = TreeNode('child1', 'decoder', tensor_path='transformer/self_attn')
    golden_grandchild1 = TreeNode('grandchild1', 'self_attn', tensor_path='transformer/layer2/self_attn')
    golden_grandchild2 = TreeNode('grandchild2', 'mlp', tensor_path='transformer/layer2/mlp')
    
    golden_child1.add_child(golden_grandchild1)
    golden_child1.add_child(golden_grandchild2)
    golden_root.add_child(golden_child1)

    my_root = TreeNode('model', 'BaichuanModel')
    my_child1 = TreeNode('child1', 'root', tensor_path='transformer/Attention')
    my_grandchild1 = TreeNode('grandchild1', 'Attention', tensor_path='transformer/layer2/Attention')
    my_grandchild2 = TreeNode('grandchild2', 'Mlp', tensor_path='transformer/layer2/Mlp')
    
    my_child1.add_child(my_grandchild1)
    my_child1.add_child(my_grandchild2)
    my_root.add_child(my_child1)
    # 执行匹配策略
    match_map = OpMatchMap(golden_root, my_root)
    policy_module_match(golden_root, my_root, match_map)
    matches = match_map.get_match_map(enable_print=True)
    expected_matches = [
        (my_grandchild1, MatchLocation.ALL_OUTPUT, golden_grandchild1, MatchLocation.ALL_OUTPUT),
        (my_grandchild2, MatchLocation.ALL_OUTPUT, golden_grandchild2, MatchLocation.ALL_OUTPUT),
    ]
    assert len(matches) == len(expected_matches)
    for match, expected in zip(matches, expected_matches):
        assert match == expected