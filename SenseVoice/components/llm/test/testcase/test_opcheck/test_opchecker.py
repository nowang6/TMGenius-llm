import unittest 
import os
import stat
from unittest.mock import patch, MagicMock
import pytest

from msit_llm.opcheck.opchecker import _is_atb_only_saved_before, OpChecker
from msit_llm.common.log import logger

@pytest.fixture()
def mock_logger():
    with patch.object(logger, 'error') as mock_error:
        yield mock_error


@pytest.mark.parametrize(
    "dump_scenario, expected_result",
    [
        (['before'], True),
        (['after'], False),
        (['before', 'after'], False)
    ]
)
def test_is_atb_only_saved_before(mock_logger, tmpdir, dump_scenario, expected_result):
    input_path = tmpdir.mkdir('token_id')
    layer_dir = input_path.mkdir('0_Decoder_layer')
    for folders in dump_scenario:
        layer_dir.join(folders).ensure()
    res = _is_atb_only_saved_before(str(input_path))

    assert res is expected_result
    mock_logger.assert_not_called()


def test_is_atb_only_saved_before_false_no_folders(mock_logger, tmpdir):
    input_path = tmpdir.mkdir('token_id')
    res = _is_atb_only_saved_before(str(input_path))

    assert res is False
    mock_logger.assert_called_once()


def third_party_init_env_path(tmp_path_factory):
    # 预置so加载环境变量
    dir_path = tmp_path_factory.mktemp("test_ait_opcheck_lib_path")
    file_path = os.path.join( dir_path / "libopchecker.so")
    os.environ['AIT_OPCHECK_LIB_PATH'] = file_path
    return file_path

def test_third_party_init_load_not_exist_file(tmp_path_factory):
    third_party_init_env_path(tmp_path_factory)

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    assert res is False


def test_third_party_init_load_other_writable_file(tmp_path_factory):
    file_path = third_party_init_env_path(tmp_path_factory)

    # 创建他人可写文件
    file_permissions = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    with os.fdopen(os.open(file_path, os.O_CREAT, file_permissions), 'w'):
        pass

    res = OpChecker().third_party_init()

    del os.environ['AIT_OPCHECK_LIB_PATH']
    if os.path.exists(file_path):
        os.remove(file_path)
    assert res is False

class TestOpChecker(unittest.TestCase):
    def setUp(self):
        self.op_checker = OpChecker()
        self.op_checker.base_path = '/mock/base/path'
        self.op_checker.operation_ids = ''
        self.op_checker.operation_name = None
        self.op_checker.check_ids_string = []
        self.op_checker.check_patterns = []
        self.op_checker.precision_metric = []
        self.op_checker.optimization_identify = False
        self.op_checker.cases_info = {}
        
    @patch('os.path.exists')
    def test_get_base_path_valid(self, mock_exists):
        mock_exists.return_value = True
        op_checker = OpChecker()
        result_path, pid = op_checker.get_base_path('/msit_dump/tensors/operation_12345/path')
        self.assertEqual(result_path, '/msit_dump/tensors/operation_12345/path')
        self.assertEqual(pid, '12345')

    @patch('os.path.exists')
    def test_get_base_path_invalid(self, mock_exists):
        mock_exists.return_value = False
        op_checker = OpChecker()
        result_path, pid = op_checker.get_base_path('/invalid/path')
        self.assertIsNone(result_path)
        self.assertIsNone(pid)

    @patch('os.path.exists')
    @patch('msit_llm.common.log.logger.error')
    def test_check_input_legality_invalid(self, mock_logger_error, mock_exists):
        mock_exists.return_value = False
        op_checker = OpChecker()
        input_path, base_path, pid, ret = op_checker.check_input_legality('/invalid/input/path')
        self.assertFalse(ret)
        self.assertEqual(input_path, '/invalid/input/path')
        self.assertIsNone(base_path)
        self.assertIsNone(pid)
        mock_logger_error.assert_called_once()

    @patch('os.path.exists')
    @patch('msit_llm.common.log.logger.error')
    @patch('msit_llm.common.log.logger.info')
    def test_args_init_success(self, mock_logger_info, mock_logger_error, mock_exists):
        mock_exists.return_value = True
        args_mock = MagicMock(
            input='/valid/input/path',
            output='/valid/output/path',
            operation_ids='id1,id2',
            operation_name='name1,name2',
            precision_metric=['abs', 'kl'],
            precision_mode='keep_origin_dtype',
            jobs=4,
            log_level='info',
            custom_algorithms=False,
            device_id=0,
            atb_rerun=True,
            optimization_identify=False
        )
        with patch.object(OpChecker, 'third_party_init', return_value=True) as mock_third_party_init:
            op_checker = OpChecker()
            execution_flag = op_checker.args_init(args_mock)
            self.assertFalse(execution_flag)
            self.assertEqual(op_checker.input, '/valid/input/path')
            self.assertEqual(op_checker.output, '/valid/output/path')
            self.assertEqual(op_checker.check_ids_string, ['id1', 'id2'])
            self.assertEqual(op_checker.check_patterns, ['name1', 'name2'])
            self.assertEqual(op_checker.precision_metric, ['abs', 'kl'])
            self.assertEqual(op_checker.jobs, 4)
            self.assertEqual(op_checker.log_level, 'info')
            self.assertFalse(op_checker.custom_algorithms)
            mock_third_party_init.assert_called_once()

    def test_check_id_range_match(self):
        op_checker = OpChecker()
        op_checker.operation_ids = '123,456'
        op_checker.check_ids_string = ['123', '456']
        self.assertTrue(op_checker.check_id_range('123_0'))

    def test_check_id_range_no_match(self):
        op_checker = OpChecker()
        op_checker.operation_ids = '123,456'
        op_checker.check_ids_string = ['123', '456']
        self.assertFalse(op_checker.check_id_range('789_0'))

    def test_check_name_match(self):
        op_checker = OpChecker()
        op_checker.operation_name = 'operation,test'
        op_checker.check_patterns = ['operation', 'test']
        self.assertTrue(op_checker.check_name('Operation'))

    def test_check_name_no_match(self):
        op_checker = OpChecker()
        op_checker.operation_name = 'operation,test'
        op_checker.check_patterns = ['operation', 'test']
        self.assertFalse(op_checker.check_name('invalid'))

    def test_is_exec_node_match(self):
        op_checker = OpChecker()
        op_checker.operation_ids = '123'
        op_checker.check_ids_string = ['123']
        op_checker.operation_name = 'operation'
        op_checker.check_patterns = ['operation']
        case_info = {'op_id': '123_0', 'op_name': 'Operation'}
        self.assertTrue(op_checker.is_exec_node(case_info))

    def test_is_exec_node_no_match(self):
        op_checker = OpChecker()
        op_checker.operation_ids = '123'
        op_checker.check_ids_string = ['123']
        op_checker.operation_name = 'operation'
        op_checker.check_patterns = ['operation']
        case_info = {'op_id': '456_0', 'op_name': 'Invalid'}
        self.assertFalse(op_checker.is_exec_node(case_info))

    @patch('copy.deepcopy')
    @patch('msit_llm.opcheck.check_case.self_attention.MaskType')
    @patch('msit_llm.opcheck.check_case.self_attention.KernelType')
    @patch('msit_llm.opcheck.check_case.self_attention.ClampType')
    def test_traverse_optimization_self_attention(self, *mocks):
        op_checker = OpChecker()
        op_checker.optimization_identify = True
        case_info = {
            "op_id": "self_attention_0",
            "op_name": "SelfAttentionOperation",
            "op_param": {"maskType": 1, "kernelType": 2, "clampType": 3}
        }
        op_checker.traverse_optimization(case_info, 'SelfAttentionOperation', 'self_attention_0')
        self.assertIn('self_attention_0_1', op_checker.cases_info)

    def test_check_id_range(self):
        self.op_checker.operation_ids = '0'
        self.op_checker.check_ids_string = ['0']

        op_id = '0_1_2'
        self.assertTrue(self.op_checker.check_id_range(op_id))

        op_id = '1_2_3'
        self.assertFalse(self.op_checker.check_id_range(op_id))

        self.op_checker.operation_ids = ''
        self.assertFalse(self.op_checker.check_id_range(None))

    def test_check_name(self):
        self.op_checker.operation_name = 'attention'
        self.op_checker.check_patterns = ['attention']

        op_name = 'SelfAttentionOperation'
        self.assertTrue(self.op_checker.check_name(op_name))

        op_name = 'ConvOperation'
        self.assertFalse(self.op_checker.check_name(op_name))

        self.op_checker.operation_name = None
        self.assertFalse(self.op_checker.check_name(None))

    def test_is_exec_node(self):
        case_info = {'op_id': '0_1_2', 'op_name': 'op_name'}

        # Both operation_ids and operation_name are unspecified
        self.assertTrue(self.op_checker.is_exec_node(case_info))

        self.op_checker.operation_ids = '0'
        self.op_checker.check_ids_string = ['0']
        self.op_checker.operation_name = 'op_name'
        self.op_checker.check_patterns = ['name']

        self.assertTrue(self.op_checker.is_exec_node(case_info))

        self.op_checker.operation_ids = '1'
        self.assertTrue(self.op_checker.is_exec_node(case_info))

    def test_add_case_to_cases(self):
        case_info = {'op_name': 'KvCacheOperation', 'op_id': '0_1_2'}
        self.op_checker.add_case_to_cases(case_info)
        self.assertIn('0_1_2', self.op_checker.cases_info)
        self.assertEqual(self.op_checker.cases_info['0_1_2']['inplace_idx'], [2])

        case_info = {'op_name': 'ReshapeAndCacheOperation', 'op_id': '0_1_2'}
        self.op_checker.add_case_to_cases(case_info)
        self.assertEqual(self.op_checker.cases_info['0_1_2']['inplace_idx'], [2, 3])

        case_info = {'op_name': 'SelfAttentionOperation', 'op_id': '0_1_2'}
        self.op_checker.add_case_to_cases(case_info)
        self.assertIn('0_1_2', self.op_checker.cases_info)

        case_info = {'op_name': 'UnknownOperation', 'op_id': '0_1_2'}
        self.op_checker.add_case_to_cases(case_info)
        self.assertIn('0_1_2', self.op_checker.cases_info)
