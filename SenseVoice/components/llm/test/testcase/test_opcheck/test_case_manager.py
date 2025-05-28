from unittest.mock import patch

import pytest
import pandas as pd

from msit_llm.opcheck.case_manager import CaseManager
from msit_llm.opcheck.check_case import OP_NAME_DICT
from msit_llm.opcheck.opchecker import NAMEDTUPLE_PRECISION_METRIC


# Mocking necessary imports and functions
class MockTestLoader:
    @staticmethod
    def getTestCaseNames(op):
        return ["test_case_1"]


class MockOP:
    def __init__(self, name, case_info):
        self.case_info = case_info

    def run(self, runner):
        pass


class MockQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get_nowait(self):
        return self.items.pop(0) if self.items else None

    def empty(self):
        return len(self.items) == 0


class MockProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)

    def join(self):
        pass


class MockPool:
    @staticmethod
    def Process(target, args):
        return MockProcess(target, args)


class MockManager:
    @staticmethod
    def Queue():
        return MockQueue()


class MockTextTestRunner:
    @staticmethod
    def run(suite):
        return unittest.TextTestResult(None, None, 0)


@pytest.fixture(scope="module", autouse=True)
def mock_dependencies():
    with patch('multiprocessing.get_context', return_value=MockPool()):
        with patch('multiprocessing.Manager', return_value=MockManager()):
            with patch('unittest.TextTestRunner', return_value=MockTextTestRunner()):
                with patch('unittest.TestLoader', return_value=MockTestLoader()):
                    yield


def test_init_given_parameters_when_valid():
    cm = CaseManager(precision_metric=[NAMEDTUPLE_PRECISION_METRIC.abs], rerun=True, optimization_identify=True,
                     output_path='./test_output')
    assert cm.precision_metric == [NAMEDTUPLE_PRECISION_METRIC.abs]
    assert cm.rerun is True
    assert cm.optimization_identify is True
    assert cm.output_path == './test_output'
    assert cm.cases == []


@pytest.mark.parametrize("case_queue_items, expected_result_queue_items", [
    ([{'op_name': 'valid_op'}], []),
    ([], [])
])
def test_excute_case_given_case_queue(case_queue_items, expected_result_queue_items,
                                      mock_dependencies):
    case_queue = MockQueue()
    result_queue = MockQueue()
    for item in case_queue_items:
        case_queue.put(item)
    with patch.dict(OP_NAME_DICT, {'valid_op': MockOP}):
        CaseManager.excute_case(case_queue, result_queue, 'info', [])
    assert case_queue.empty()
    assert result_queue.items == expected_result_queue_items


@pytest.mark.parametrize("op_info, res_detail, expected_result", [
    ({'op_id': '1', 'op_name': 'valid_op', 'op_param': {}, 'tensor_path': '', 'excuted_information': '',
      'fail_reason': '', 'optimization_closed': ''},
     {'precision_standard': 'standard', 'rel_pass_rate': '90%', 'max_rel': '0.1'},
     {'out_tensor_id': '1', 'precision_standard': 'standard', 'rel_precision_rate(%)': '90%', 'max_rel_error': '0.1'}),
    ({'op_id': '1', 'op_name': 'valid_op', 'op_param': {}, 'tensor_path': '', 'excuted_information': '',
      'fail_reason': '', 'optimization_closed': ''}, {},
     {'out_tensor_id': '1', 'precision_standard': 'NaN', 'rel_precision_rate(%)': 'NaN', 'max_rel_error': 'NaN'})
])
def test_update_single_op_result_given_valid_op_info(op_info, res_detail, expected_result, mock_dependencies):
    cm = CaseManager(precision_metric=[NAMEDTUPLE_PRECISION_METRIC.abs])
    updated_op_info = cm._update_single_op_result(op_info, '1', res_detail)
    for key, value in expected_result.items():
        assert updated_op_info[key] == value


@pytest.mark.parametrize("op_name, expected_result", [
    ("valid_op", True),
    ("invalid_op", False),
    ("nonexistent_op", False)
])
def test_add_case_given_case_info_when_op_name_exists(op_name, expected_result):
    cm = CaseManager(precision_metric=[], rerun=False, optimization_identify=False, output_path='./')
    case_info = {'op_name': op_name}

    with patch.dict(OP_NAME_DICT, {"valid_op": True, "invalid_op": False}):
        result = cm.add_case(case_info)

    assert result == expected_result
    if expected_result:
        assert len(cm.cases) == 1
        assert cm.cases[0] == case_info
    else:
        assert len(cm.cases) == 0


@pytest.mark.parametrize("num_processes, rerun, log_level, custom_algorithms", [
    (1, False, 'info', []),
    (2, False, 'debug', ['custom_alg']),
    (1, True, 'info', []),
    (2, True, 'debug', ['custom_alg'])
])
def test_excute_cases_given_valid_cases(num_processes, rerun, log_level, custom_algorithms,
                                        mock_dependencies):
    cm = CaseManager(precision_metric=[], rerun=rerun, optimization_identify=False, output_path='./')
    case_info = {'op_name': 'valid_op'}
    cm.add_case(case_info)

    with patch.dict(OP_NAME_DICT, {'valid_op': MockOP}):
        with patch.object(cm, 'single_process') as mock_single_process:
            with patch.object(cm, 'multi_process') as mock_multi_process:
                cm.excute_cases(num_processes=num_processes, log_level=log_level, custom_algorithms=custom_algorithms)

    if num_processes == 1 or rerun:
        mock_single_process.assert_called_once()
    else:
        mock_multi_process.assert_called_once_with(num_processes, log_level, custom_algorithms)


@pytest.mark.parametrize("results, expected_call_count", [
    ([{'op_id': '1', 'op_name': 'valid_op', 'op_param': {}, 'tensor_path': '', 'excuted_information': '',
       'fail_reason': '', 'optimization_closed': '', 'res_detail': []}], 1),
    ([], 0)
])
def test_write_op_result_to_csv_given_valid_results(results, expected_call_count,
                                                    mock_dependencies):
    cm = CaseManager(precision_metric=[], rerun=False, optimization_identify=False, output_path='./test_output.xlsx')

    with patch('pandas.DataFrame.to_excel') as mock_to_excel:
        cm.write_op_result_to_csv(results)

    assert mock_to_excel.call_count == expected_call_count
