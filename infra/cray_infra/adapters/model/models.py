"""
ScalarLM-specific model adapters and extensions for vLLM models.
"""

import torch
from torch import nn
from typing import Optional, Any, Dict, List, Type, ClassVar, Literal
import logging

from ..common.interfaces import SupportsTokenformer
from .tokenformer import TokenformerManager as TokenformerModelManager, create_tokenformer_manager

logger = logging.getLogger(__name__)


class ScalarLMModelMixin:
    """Mixin class that provides ScalarLM-specific functionality to vLLM models."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scalarlm_initialized = False
        self._tokenformer_manager: Optional[TokenformerModelManager] = None
        
    def initialize_scalarlm_features(self, device: torch.device):
        """Initialize ScalarLM-specific features."""
        if self._scalarlm_initialized:
            return
            
        logger.info("Initializing ScalarLM features")
        
        # Initialize Tokenformer if supported
        if hasattr(self, 'supports_tokenformer') and self.supports_tokenformer:
            self._tokenformer_manager = create_tokenformer_manager(self, device)
            logger.info("Tokenformer manager initialized")
            
        self._scalarlm_initialized = True
        
    def get_tokenformer_manager(self) -> Optional[TokenformerModelManager]:
        """Get the Tokenformer manager for this model."""
        return self._tokenformer_manager


class ScalarLMTokenformerModel(ScalarLMModelMixin, SupportsTokenformer):
    """Base class for ScalarLM models that support Tokenformer."""
    
    supports_tokenformer: ClassVar[Literal[True]] = True
    
    def enable_tokenformer(self, config: dict) -> None:
        """Enable Tokenformer functionality for this model."""
        if not self._scalarlm_initialized:
            # Try to determine device from model parameters
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cuda')
            self.initialize_scalarlm_features(device)
            
        if self._tokenformer_manager:
            logger.info("Enabling Tokenformer functionality")
            # Additional Tokenformer configuration can be applied here
        else:
            logger.warning("Tokenformer manager not available")
            
    def disable_tokenformer(self) -> None:
        """Disable Tokenformer functionality for this model."""
        if self._tokenformer_manager and self._tokenformer_manager.get_active_adapter():
            active_adapter = self._tokenformer_manager.get_active_adapter()
            self._tokenformer_manager.deactivate_adapter(active_adapter)
            logger.info("Tokenformer functionality disabled")


def create_scalarlm_model_adapter(base_model_class: Type) -> Type:
    """
    Create a ScalarLM adapter for an existing vLLM model class.
    
    Args:
        base_model_class: The base vLLM model class to extend
        
    Returns:
        A new class that combines the base model with ScalarLM features
    """
    
    class ScalarLMAdaptedModel(ScalarLMTokenformerModel, base_model_class):
        """Dynamically created ScalarLM adapter model."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize ScalarLM features after base model initialization
            if hasattr(self, 'device'):
                device = self.device
            else:
                device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda')
                
            self.initialize_scalarlm_features(device)
    
    # Set appropriate class name and module
    ScalarLMAdaptedModel.__name__ = f"ScalarLM{base_model_class.__name__}"
    ScalarLMAdaptedModel.__qualname__ = f"ScalarLM{base_model_class.__qualname__}"
    
    return ScalarLMAdaptedModel


def register_scalarlm_model_adapters():
    """Register ScalarLM model adapters with common vLLM models."""
    try:
        # Import common vLLM models
        from vllm.model_executor.models.llama import LlamaForCausalLM
        from vllm.model_executor.models.gemma import GemmaForCausalLM
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
        
        from ..vllm.registry import get_registry
        
        registry = get_registry()
        
        # Create and register ScalarLM adapters
        models_to_adapt = [
            ("scalarlm_llama", LlamaForCausalLM),
            ("scalarlm_gemma", GemmaForCausalLM), 
            ("scalarlm_qwen2", Qwen2ForCausalLM),
        ]
        
        for model_name, base_class in models_to_adapt:
            try:
                adapted_class = create_scalarlm_model_adapter(base_class)
                registry.register_model(
                    model_name, 
                    adapted_class,
                    {"supports_tokenformer": True}
                )
                logger.info(f"Registered ScalarLM adapter: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to register {model_name}: {e}")
                
    except ImportError as e:
        logger.warning(f"Could not import vLLM models for adaptation: {e}")
    except Exception as e:
        logger.error(f"Error registering ScalarLM model adapters: {e}")


# Model factory functions
def create_scalarlm_llama_model(*args, **kwargs):
    """Create a ScalarLM-adapted Llama model."""
    from vllm.model_executor.models.llama import LlamaForCausalLM
    ScalarLMLlama = create_scalarlm_model_adapter(LlamaForCausalLM)
    return ScalarLMLlama(*args, **kwargs)


def create_scalarlm_gemma_model(*args, **kwargs):
    """Create a ScalarLM-adapted Gemma model.""" 
    from vllm.model_executor.models.gemma import GemmaForCausalLM
    ScalarLMGemma = create_scalarlm_model_adapter(GemmaForCausalLM)
    return ScalarLMGemma(*args, **kwargs)


def create_scalarlm_qwen2_model(*args, **kwargs):
    """Create a ScalarLM-adapted Qwen2 model."""
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    ScalarLMQwen2 = create_scalarlm_model_adapter(Qwen2ForCausalLM)
    return ScalarLMQwen2(*args, **kwargs)