"""
Factory for creating vLLM engines based on configuration.
Supports both HTTP and direct engine modes.
"""

import logging
from typing import Optional

from ..util.get_config import get_config
from .engine_interface import VLLMEngineInterface
from .http_engine import HTTPVLLMEngine
from .direct_engine import DirectVLLMEngine
from .shared_engine import SharedVLLMEngine

logger = logging.getLogger(__name__)


def create_vllm_engine(config: Optional[dict] = None) -> VLLMEngineInterface:
    """
    Create a vLLM engine based on configuration.
    
    Args:
        config: Configuration dictionary. If None, uses get_config()
        
    Returns:
        VLLMEngineInterface instance (HTTP, Direct, or Shared)
        
    Raises:
        RuntimeError: If engine creation fails
    """
    if config is None:
        config = get_config()
    
    # Determine engine type from configuration
    engine_type = config.get("vllm_engine_type", "http")  # http, direct, shared
    
    # Legacy support for vllm_use_http
    if "vllm_use_http" in config and "vllm_engine_type" not in config:
        engine_type = "http" if config["vllm_use_http"] else "direct"
    
    if engine_type == "http":
        return _create_http_engine(config)
    elif engine_type == "shared":
        return _create_shared_engine(config)
    else:  # direct
        return _create_direct_engine(config)


def _create_http_engine(config: dict) -> HTTPVLLMEngine:
    """Create HTTP-based vLLM engine."""
    vllm_api_url = config.get("vllm_api_url", "http://localhost:8001")
    timeout = config.get("vllm_http_timeout", 30.0)
    
    logger.info(f"Creating HTTP vLLM engine: {vllm_api_url}")
    
    try:
        engine = HTTPVLLMEngine(
            base_url=vllm_api_url,
            timeout=timeout
        )
        logger.info("✓ HTTP vLLM engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create HTTP vLLM engine: {e}")
        raise RuntimeError(f"HTTP vLLM engine creation failed: {e}")


def _create_direct_engine(config: dict) -> DirectVLLMEngine:
    """Create direct vLLM engine."""
    logger.info("Creating direct vLLM engine")
    
    try:
        # Import vLLM components
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.usage.usage_lib import UsageContext
        from vllm.config import ModelConfig
        
        # Handle model-specific configurations
        model_name = config.get("model", "microsoft/DialoGPT-small")
        
        # Disable LoRA for GPT-2 models as they don't support it
        enable_lora = config.get("enable_lora", True)
        if "gpt2" in model_name.lower() or "dialogpt" in model_name.lower():
            enable_lora = False
            logger.info(f"Disabling LoRA for GPT-2 model: {model_name}")
        
        # Create engine arguments from config
        engine_args = AsyncEngineArgs(
            model=model_name,
            dtype=config.get("dtype", "auto"),
            max_model_len=config.get("max_model_length", 2048),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            enable_lora=enable_lora,
            max_lora_rank=config.get("max_lora_rank", 16),
            # Add other relevant parameters
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            pipeline_parallel_size=config.get("pipeline_parallel_size", 1),
            trust_remote_code=config.get("trust_remote_code", False),
            download_dir=config.get("download_dir"),
            load_format=config.get("load_format", "auto"),
            quantization=config.get("quantization"),
            served_model_name=config.get("served_model_name"),
            revision=config.get("revision"),
            code_revision=config.get("code_revision"),
            rope_theta=config.get("rope_theta"),
            tokenizer_revision=config.get("tokenizer_revision"),
            max_cpu_loras=config.get("max_cpu_loras"),
            disable_log_stats=config.get("disable_log_stats", False),
            enforce_eager=config.get("enforce_eager", False),
            max_seq_len_to_capture=config.get("max_seq_len_to_capture", 8192),
            disable_custom_all_reduce=config.get("disable_custom_all_reduce", False),
        )
        
        # Create the async engine
        logger.info("Initializing AsyncLLMEngine...")
        engine = AsyncLLMEngine.from_engine_args(
            engine_args, 
            usage_context=UsageContext.API_SERVER
        )
        
        # Create model config for serving layers
        model_config = ModelConfig(
            model=model_name,
            task="generate",
            tokenizer=config.get("tokenizer"),
            tokenizer_mode=config.get("tokenizer_mode", "auto"),
            trust_remote_code=config.get("trust_remote_code", False),
            dtype=config.get("dtype", "auto"),
            seed=config.get("seed", 0),
            revision=config.get("revision"),
            code_revision=config.get("code_revision"),
            rope_theta=config.get("rope_theta"),
            tokenizer_revision=config.get("tokenizer_revision"),
            max_model_len=config.get("max_model_length", 2048),
            quantization=config.get("quantization"),
            quantization_param_path=config.get("quantization_param_path"),
            enforce_eager=config.get("enforce_eager", False),
            max_seq_len_to_capture=config.get("max_seq_len_to_capture", 8192),
            max_logprobs=config.get("max_logprobs", 20),
            disable_sliding_window=config.get("disable_sliding_window", False),
            skip_tokenizer_init=config.get("skip_tokenizer_init", False),
            served_model_name=config.get("served_model_name"),
        )
        
        # Create direct engine wrapper
        direct_engine = DirectVLLMEngine(engine, model_config)
        
        logger.info("✓ Direct vLLM engine created successfully")
        return direct_engine
        
    except ImportError as e:
        logger.error(f"vLLM imports failed: {e}")
        raise RuntimeError(f"vLLM components not available for direct engine: {e}")
    except Exception as e:
        logger.error(f"Failed to create direct vLLM engine: {e}")
        raise RuntimeError(f"Direct vLLM engine creation failed: {e}")


def _create_shared_engine(config: dict) -> SharedVLLMEngine:
    """Create shared vLLM engine that connects to existing engine instance."""
    logger.info("Creating shared vLLM engine")
    
    try:
        engine_name = config.get("vllm_shared_engine_name", "default")
        
        # Create shared engine wrapper  
        shared_engine = SharedVLLMEngine(engine_name=engine_name)
        
        logger.info(f"✓ Shared vLLM engine created successfully: {engine_name}")
        return shared_engine
        
    except Exception as e:
        logger.error(f"Failed to create shared vLLM engine: {e}")
        raise RuntimeError(f"Shared vLLM engine creation failed: {e}")


async def test_engine(engine: VLLMEngineInterface) -> bool:
    """
    Test if an engine is working correctly.
    
    Args:
        engine: Engine to test
        
    Returns:
        True if engine is working, False otherwise
    """
    try:
        logger.info(f"Testing {engine.engine_type} engine...")
        
        # Test health check
        health_ok = await engine.health_check()
        if not health_ok:
            logger.warning(f"{engine.engine_type} engine health check failed")
            return False
        
        # Test embedding generation (if supported)
        try:
            test_embedding = await engine.generate_embeddings(
                "test prompt", 
                "default"
            )
            if test_embedding and len(test_embedding) > 0:
                logger.info(f"✓ {engine.engine_type} engine embedding test passed")
            else:
                logger.warning(f"{engine.engine_type} engine embedding test returned empty result")
        except Exception as e:
            logger.warning(f"{engine.engine_type} engine embedding test failed: {e}")
        
        # Test completion generation (if supported)
        try:
            test_completion = await engine.generate_completion(
                "Hello", 
                "default",
                max_tokens=5
            )
            if test_completion:
                logger.info(f"✓ {engine.engine_type} engine completion test passed")
            else:
                logger.warning(f"{engine.engine_type} engine completion test returned empty result")
        except Exception as e:
            logger.warning(f"{engine.engine_type} engine completion test failed: {e}")
        
        logger.info(f"✓ {engine.engine_type} engine test completed")
        return True
        
    except Exception as e:
        logger.error(f"{engine.engine_type} engine test failed: {e}")
        return False


def get_engine_info(engine: VLLMEngineInterface) -> dict:
    """
    Get information about the engine.
    
    Args:
        engine: Engine to inspect
        
    Returns:
        Dictionary with engine information
    """
    return {
        "type": engine.engine_type,
        "class": type(engine).__name__,
        "repr": str(engine)
    }