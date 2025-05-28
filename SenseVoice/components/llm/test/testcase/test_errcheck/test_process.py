import os
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

from msit_llm.errcheck.process import handles_so_dir, handles_check_type, \
                                 handles_exec, handles_exit_flag, handles_output_dir


class TestErrorCheck(TestCase):
    
    def setUp(self) -> None:
        self.args = MagicMock()
    
    def test_check_so_dir_should_raise_when_cann_dir_empty(self):
        os.environ["ASCEND_TOOLKIT_HOME"] = ""
        
        with self.assertRaises(OSError):
            handles_so_dir()
    
    def test_check_so_dir_should_raise_when_cann_dir_not_exist(self):
        os.environ["ASCEND_TOOLKIT_HOME"] = "some/arbitrary/directories"
        
        with self.assertRaises(OSError):
            handles_so_dir()
    
    def test_check_so_dir_should_raise_when_so_not_found(self):
        os.environ["ASCEND_TOOLKIT_HOME"] = "/usr/local"
        
        with self.assertRaises(OSError):
            handles_so_dir()
    
    # If ASCEND_TOOLKIT_HOME is not set, `handles_so_dir()` will raise error if so is not found,
    # or, it will not raise anything if so is founded under the default directory
    
    def test_handles_exec_should_raise_when_subcommand_empty(self):
        self.args.exec = ""
        
        with self.assertRaises(ValueError):
            handles_exec(self.args)
    
    def test_handles_exec_should_raise_when_subcommand_are_spaces(self):
        self.args.exec = "         "
        
        with self.assertRaises(ValueError):
            handles_exec(self.args)
    
    def test_handles_check_type_should_1_when_specify_overflow(self):
        self.args.type = ["overflow"]
        
        handles_check_type(self.args)
        
        self.assertEqual(os.environ["ATB_CHECK_TYPE"], "1")
        
    # Currently, check type only supports overflow
    # there may be other features in the future
    # if so, add the test here. 
    # E.g.
    # def test_handles_check_type_should_2_when_specify_memleak(self):
    # self.args.type = ["memleak"]
    
    def test_handles_output_dir_should_equal_current_dir_when_empty_output_dir(self):
        self.args.output = ""
        
        handles_output_dir(self.args)
        
        self.assertEqual(os.environ["ATB_OUTPUT_DIR"], os.getcwd())
        
    def test_handles_output_dir_should_valid_when_specify_arbitrary_dir(self):
        self.args.output = "./abc/def"
        
        handles_output_dir(self.args)
        
        temp_dir = os.environ["ATB_OUTPUT_DIR"]
        self.assertTrue(os.path.abspath(temp_dir), os.getcwd() + temp_dir)
    
    def test_handles_exit_flag_should_1_when_specify(self):
        self.args.exit = True
        
        handles_exit_flag(self.args)
        
        self.assertEqual(os.environ["ATB_EXIT"], "1")
        
    def test_handles_exit_flag_should_0_when_not_specify(self):
        self.args.exit = False
        
        handles_exit_flag(self.args)
        
        self.assertEqual(os.environ["ATB_EXIT"], "0")
        

if __name__ == "__main__":
    unittest.main()