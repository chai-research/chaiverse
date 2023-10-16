import unittest
import importlib

def is_peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None

def require_peft(test_case):
    """
    Decorator marking a test that requires peft. Skips the test if peft is not available.
    """
    if not is_peft_available():
        test_case = unittest.skip("test requires peft")(test_case)
    return test_case

