from unittest.mock import MagicMock

from msit_llm.opcheck.check_case.split import OpcheckAddOperation


# Mocking the OperationTest class to avoid errors
class MockOperationTest:
    def __init__(self, *args, **kwargs):
        self.op_param = kwargs.get('op_param', {})

    @staticmethod
    def validate_param(self, *args, **kwargs):
        # Mock validation; return True for simplicity
        return True

    def execute(self):
        # Mock execution; do nothing
        pass

# Replace the base class with the mock
OpcheckAddOperation.__bases__ = (MockOperationTest,)

def test_test_method_given_valid_params_when_called_then_execute_called():
    # Arrange
    op = OpcheckAddOperation(op_param={'splitDim': 0, 'splitNum': 2})
    op.execute = MagicMock()
    
    # Act
    op.test()
    
    # Assert
    op.execute.assert_called_once()
