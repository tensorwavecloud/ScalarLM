"""
ScalarLM Adapters - Clean Architecture

This package provides adapters to integrate ScalarLM-specific functionality 
with various ML frameworks in a decoupled way.

Structure:
- vllm/: vLLM-specific adapters and extensions
- model/: Model-agnostic adapters (Tokenformer, etc.)
- common/: Common adapter utilities and interfaces

## Clean Architecture Principles:
1. vLLM imports NO ScalarLM code
2. ScalarLM adapters inject functionality into vLLM
3. Configuration is provided via dependency injection
4. All ScalarLM-specific code is in this adapter package
5. ScalarLM REQUIRES the vLLM fork for optimal compatibility

## Usage:
```python

# Create regular vLLM model
vllm_model = LLM(model="meta-llama/Llama-2-7b-hf")

# Enhance with ScalarLM functionality (no vLLM code changes needed)
enhanced_model = enhance_vllm_model(vllm_model.llm_engine.model_executor, device="cuda")
```
"""

__version__ = "0.2.0"

# ScalarLM Adapter Exports
from .vllm.adapter import (
    init_adapters,  # Primary initialization function
    enhance_vllm_model,
    create_tokenformer_manager,
    enhanced_vllm,  # Primary context manager
    scalarlm_enhanced_vllm,  # Deprecated alias
    AdapterManager,
    EnhancedModelWrapper,
    ScalarLMAdapter,
    ScalarLMConfigProvider,
)

# Tokenformer Components
from .model.tokenformer import (
    TokenformerModel,
    TokenformerManager,
    create_tokenformer_manager,
    default_supports_tokenformer_check,
)

# Common Adapter Utilities
from .common.adapter_commons import (
    AdapterModel,
    AdapterModelManager,
    ConfigProvider,
    get_config_provider,
    set_config_provider,
)

# Attention Components
from .vllm.attention_adapter import (
    AttentionType,
    AttentionMetadataAdapter,
    VLLMAttentionAdapter,
    create_attention_adapter,
    patch_vllm_attention_layer,
)

# Legacy interfaces (kept for compatibility)
from .common.interfaces import SupportsTokenformer, supports_tokenformer

# Migration utilities (currently unused but kept for potential future use)
# from .vllm.registry import register_scalarlm_models, get_scalarlm_model_class

__all__ = [
    # Primary API
    "init_adapters",  # Primary initialization function
    "enhance_vllm_model", 
    "create_tokenformer_manager",
    "enhanced_vllm",  # Primary context manager
    "scalarlm_enhanced_vllm",  # Deprecated alias
    
    # Core Components
    "AdapterManager",
    "EnhancedModelWrapper", 
    "ScalarLMAdapter",
    "ScalarLMConfigProvider",
    
    # Tokenformer
    "TokenformerModel",
    "TokenformerManager",
    "create_tokenformer_manager",
    "default_supports_tokenformer_check",
    
    # Adapter Commons
    "AdapterModel",
    "AdapterModelManager", 
    "ConfigProvider",
    "get_config_provider",
    "set_config_provider",
    
    # Attention
    "AttentionType",
    "AttentionMetadataAdapter",
    "VLLMAttentionAdapter",
    "create_attention_adapter", 
    "patch_vllm_attention_layer",
    
    # Legacy interfaces (kept for compatibility)
    "SupportsTokenformer",
    "supports_tokenformer",
    
    # Migration utilities (commented out - currently unused)
    # "register_scalarlm_models",
    # "get_scalarlm_model_class",
]
