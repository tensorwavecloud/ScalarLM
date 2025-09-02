"""
vLLM integration module for ScalarLM.
Provides both HTTP and direct engine access patterns.
"""

from .engine_interface import VLLMEngineInterface
from .http_engine import HTTPVLLMEngine
from .direct_engine import DirectVLLMEngine
from .engine_factory import create_vllm_engine, test_engine, get_engine_info

__all__ = [
    "VLLMEngineInterface",
    "HTTPVLLMEngine", 
    "DirectVLLMEngine",
    "create_vllm_engine",
    "test_engine",
    "get_engine_info",
]