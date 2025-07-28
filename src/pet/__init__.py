"""
Prompt Engineering Toolkit (PET)

A comprehensive toolkit for testing and evaluating LLM safety through red teaming prompts.
"""

__version__ = "1.0.0"
__author__ = "Wolfgang Dremmler"
__email__ = "wolfgang@example.com"

from .core.prompt_manager import PromptManager
from .core.evaluator import SafetyEvaluator
from .core.test_runner import TestRunner
from .core.config import Config

__all__ = [
    "PromptManager",
    "SafetyEvaluator", 
    "TestRunner",
    "Config",
]