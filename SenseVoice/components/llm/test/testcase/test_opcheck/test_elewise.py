from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.elewise import OpcheckElewiseAddOperation, ElewiseType
from mock_operation_test import MockOperationTest

# 使用新的 OperationTest 类替换原始的 OperationTest
OpcheckElewiseAddOperation.__bases__ = (MockOperationTest,)


@pytest.mark.parametrize("elewise_type, op_param, in_tensors, expected_result", [
    (ElewiseType.ELEWISE_MULS.value, {'varAttr': 2.0}, [torch.tensor([1.0, 2.0])], [torch.tensor([2.0, 4.0])]),
    (ElewiseType.ELEWISE_COS.value, {}, [torch.tensor([0.0, 1.0])], [torch.tensor([1.0, 0.5403])]),
    (ElewiseType.ELEWISE_SIN.value, {}, [torch.tensor([0.0, 1.0])], [torch.tensor([0.0, 0.8415])]),
    (ElewiseType.ELEWISE_NEG.value, {}, [torch.tensor([1.0, -2.0])], [torch.tensor([-1.0, 2.0])]),
    (ElewiseType.ELEWISE_QUANT.value, {}, [torch.tensor([1.0, 2.0])], [torch.tensor([1, 2], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_LOGICAL_NOT.value, {}, [torch.tensor([True, False])], [torch.tensor([False, True])]),
    (ElewiseType.ELEWISE_ADD.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([4.0, 6.0])]),
    (ElewiseType.ELEWISE_MUL.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
     [torch.tensor([3.0, 8.0])]),
    (ElewiseType.ELEWISE_REALDIV.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])],
     [torch.tensor([1.0, 1.0])]),
    (ElewiseType.ELEWISE_LOGICAL_AND.value, {}, [torch.tensor([True, False]), torch.tensor([True, True])],
     [torch.tensor([1, 0], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_LOGICAL_OR.value, {}, [torch.tensor([True, False]), torch.tensor([False, False])],
     [torch.tensor([1, 0], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_LESS.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
     [torch.tensor([1, 0], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_GREATER.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
     [torch.tensor([0, 1], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_SUB.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])],
     [torch.tensor([-1.0, 1.0])]),
    (ElewiseType.ELEWISE_EQUAL.value, {}, [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0])],
     [torch.tensor([1, 0], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value, {},
     [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
     [torch.tensor([1, 2], dtype=torch.int8)]),
    (ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value, {},
     [torch.tensor([1, 2], dtype=torch.int8), torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0])],
     [torch.tensor([1.0, 2.0], dtype=torch.float16)]),
    (ElewiseType.ELEWISE_TANH.value, {}, [torch.tensor([0.0, 1.0])], [torch.tensor([0.0, 0.7616])]),
])
def test_golden_calc_given_elewise_type_op_param_in_tensors_when_valid_input_then_correct_result(elewise_type, op_param,
                                                                                                 in_tensors,
                                                                                                 expected_result):
    # Arrange
    op = OpcheckElewiseAddOperation()
    op.op_param = {'elewiseType': elewise_type, **op_param}

    # Act
    result = op.golden_calc(in_tensors)

    # Assert
    for res, exp in zip(result, expected_result):
        assert torch.allclose(res, exp, atol=1e-4)


@pytest.mark.parametrize("op_param, validate_param_return, expected_execute_call", [
    ({'elewiseType': ElewiseType.ELEWISE_CAST.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_MULS.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_COS.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_SIN.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_NEG.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_QUANT.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_NOT.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_ADD.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_MUL.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_REALDIV.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_AND.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_LOGICAL_OR.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_LESS.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_GREATER.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_SUB.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_EQUAL.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_QUANT_PER_CHANNEL.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_DEQUANT_PER_CHANNEL.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_DYNAMIC_QUANT.value}, True, True),
    ({'elewiseType': ElewiseType.ELEWISE_TANH.value}, True, True),
    ({}, False, False),
    ({'elewiseType': ElewiseType.ELEWISE_UNDEFINED.value}, False, False),
])
def test_test_given_op_param_when_valid_input_then_execute_successfully(op_param, validate_param_return,
                                                                        expected_execute_call):
    # Arrange
    op = OpcheckElewiseAddOperation()
    op.op_param = op_param

    # Act
    with patch.object(op, 'validate_param', return_value=validate_param_return):
        with patch.object(op, 'execute') as mock_execute:
            op.test()

    # Assert
    if expected_execute_call:
        mock_execute.assert_called_once()
    else:
        mock_execute.assert_not_called()
