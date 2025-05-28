from unittest.mock import patch

import pytest
import torch

from msit_llm.opcheck.check_case.reshape_and_cache import OpcheckReshapeAndCacheOperation, CompressType
from msit_llm.common.log import logger
from mock_operation_test import MockOperationTest

# Use the new OperationTest class to replace the original OperationTest
OpcheckReshapeAndCacheOperation.__bases__ = (MockOperationTest,)