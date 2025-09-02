"""
Model registration system for ScalarLM vLLM integration.
"""

from typing import Dict, Type, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ScalarLMModelRegistry:
    """Registry for ScalarLM-extended vLLM models."""
    
    def __init__(self):
        self._models: Dict[str, Type] = {}
        self._adapters: Dict[str, Any] = {}
        
    def register_model(self, model_name: str, model_class: Type, adapter_config: Optional[dict] = None):
        """Register a ScalarLM model class with optional adapter configuration."""
        self._models[model_name] = model_class
        if adapter_config:
            self._adapters[model_name] = adapter_config
        logger.info(f"Registered ScalarLM model: {model_name}")
        
    def get_model_class(self, model_name: str) -> Optional[Type]:
        """Get the model class for a given model name."""
        return self._models.get(model_name)
        
    def get_adapter_config(self, model_name: str) -> Optional[dict]:
        """Get the adapter configuration for a given model name.""" 
        return self._adapters.get(model_name)
        
    def list_models(self) -> list:
        """List all registered model names."""
        return list(self._models.keys())
        
    def is_registered(self, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in self._models


# Global registry instance
_registry = ScalarLMModelRegistry()


def register_scalarlm_models():
    """Register all ScalarLM-specific models.""" 
    logger.info("Registering ScalarLM models...")
    
    # Register model adapters
    from ..model.models import register_scalarlm_model_adapters
    register_scalarlm_model_adapters()
    
    # Register factory functions for easy model creation
    from ..model.models import (
        create_scalarlm_llama_model,
        create_scalarlm_gemma_model, 
        create_scalarlm_qwen2_model
    )
    
    _registry.register_model(
        "scalarlm_llama_factory", 
        create_scalarlm_llama_model,
        {"supports_tokenformer": True, "type": "factory"}
    )
    
    _registry.register_model(
        "scalarlm_gemma_factory",
        create_scalarlm_gemma_model, 
        {"supports_tokenformer": True, "type": "factory"}
    )
    
    _registry.register_model(
        "scalarlm_qwen2_factory",
        create_scalarlm_qwen2_model,
        {"supports_tokenformer": True, "type": "factory"}
    )
    
    logger.info(f"Registered {len(_registry.list_models())} ScalarLM models")
    

def get_scalarlm_model_class(model_name: str) -> Optional[Type]:
    """Get a ScalarLM model class by name."""
    return _registry.get_model_class(model_name)


def get_registry() -> ScalarLMModelRegistry:
    """Get the global model registry."""
    return _registry