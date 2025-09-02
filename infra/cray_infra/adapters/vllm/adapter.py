"""
Clean integration layer with complete dependency inversion.
vLLM has ZERO knowledge of ScalarLM - all coupling is eliminated.
"""

import logging
from typing import Any, Dict, Optional, Callable, Type, Protocol
import torch
from contextlib import contextmanager

from ..common.adapter_commons import set_config_provider, ConfigProvider
from ..model.tokenformer import create_tokenformer_manager, default_supports_tokenformer_check
from .attention_adapter import patch_vllm_attention_layer

logger = logging.getLogger(__name__)


class VLLMModelProtocol(Protocol):
    """Protocol for vLLM models - no concrete dependency."""
    def named_modules(self): ...
    def named_parameters(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict: dict, strict: bool = True): ...
    def parameters(self): ...


class ScalarLMConfigProvider(ConfigProvider):
    """ScalarLM-specific configuration provider."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {}
        self._scalarlm_config_loaded = False
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if key in self._config:
            return self._config[key]
            
        # Lazy load ScalarLM config only when needed
        if not self._scalarlm_config_loaded:
            self._load_scalarlm_config()
            
        return self._config.get(key, default)
        
    def _load_scalarlm_config(self):
        """Load ScalarLM configuration if available."""
        try:
            # Import ScalarLM config system
            from cray_infra.util.get_config import get_config
            scalarlm_config = get_config()
            
            # Extract relevant values
            self._config.update({
                "tokenformer_cache_capacity": getattr(scalarlm_config, 'tokenformer_cache_capacity', 4),
                "enable_tokenformer": getattr(scalarlm_config, 'enable_tokenformer', True),
                "auto_patch_attention": getattr(scalarlm_config, 'auto_patch_attention', True),
            })
            
            logger.info("Loaded ScalarLM configuration")
            
        except ImportError:
            logger.info("ScalarLM config not available, using defaults")
            self._config.update({
                "tokenformer_cache_capacity": 4,
                "enable_tokenformer": True,
                "auto_patch_attention": True,
            })
            
        self._scalarlm_config_loaded = True


class ScalarLMAdapter:
    """
    Main adapter that provides ScalarLM functionality to vLLM models.
    Uses dependency injection and protocols to eliminate coupling.
    """
    
    def __init__(
        self,
        config_provider: Optional[ConfigProvider] = None,
        supports_tokenformer_check: Optional[Callable] = None
    ):
        self.config_provider = config_provider or ScalarLMConfigProvider()
        self.supports_tokenformer_check = supports_tokenformer_check or default_supports_tokenformer_check
        self._enhanced_models: Dict[str, Any] = {}
        
        # Set global config provider
        set_config_provider(self.config_provider)
        
    def enhance_vllm_model(
        self, 
        model: VLLMModelProtocol, 
        device: torch.device,
        model_id: Optional[str] = None
    ) -> "EnhancedModelWrapper":
        """
        Enhance a vLLM model with ScalarLM functionality.
        Returns a wrapper that provides ScalarLM features.
        """
        model_id = model_id or f"model_{id(model)}"
        
        if model_id in self._enhanced_models:
            return self._enhanced_models[model_id]
            
        logger.info(f"Enhancing vLLM model {model_id} with ScalarLM functionality")
        
        # Create enhanced wrapper
        wrapper = EnhancedModelWrapper(
            model=model,
            device=device,
            config_provider=self.config_provider,
            supports_tokenformer_check=self.supports_tokenformer_check
        )
        
        self._enhanced_models[model_id] = wrapper
        return wrapper
        
    def create_tokenformer_manager(self, model: VLLMModelProtocol, device: torch.device):
        """Create a tokenformer manager for a model."""
        return create_tokenformer_manager(
            model=model,
            device=device,
            supports_tokenformer_check=self.supports_tokenformer_check
        )


class EnhancedModelWrapper:
    """
    Wrapper that provides ScalarLM functionality around a vLLM model.
    The vLLM model remains completely unaware of ScalarLM.
    """
    
    def __init__(
        self,
        model: VLLMModelProtocol,
        device: torch.device, 
        config_provider: ConfigProvider,
        supports_tokenformer_check: Callable
    ):
        self.vllm_model = model
        self.device = device
        self.config_provider = config_provider
        self._tokenformer_manager = None
        self._attention_patched = False
        
        # Check if model supports tokenformer
        self.supports_tokenformer = supports_tokenformer_check(model)
        
        if self.supports_tokenformer:
            self._initialize_tokenformer()
            
    def _initialize_tokenformer(self):
        """Initialize tokenformer functionality."""
        logger.info("Initializing tokenformer for enhanced model")
        
        # Create tokenformer manager
        self._tokenformer_manager = create_tokenformer_manager(
            model=self.vllm_model,
            device=self.device,
            capacity=self.config_provider.get_config_value("tokenformer_cache_capacity", 4),
            supports_tokenformer_check=lambda m: True  # We already checked
        )
        
        # Patch attention layers if enabled
        if self.config_provider.get_config_value("auto_patch_attention", True):
            self._patch_attention_layers()
            
    def _patch_attention_layers(self):
        """Patch attention layers for tokenformer support."""
        if self._attention_patched:
            return
            
        config = {"enable_tokenformer": True}
        
        for name, module in self.vllm_model.named_modules():
            if self._is_attention_layer(name):
                logger.debug(f"Patching attention layer: {name}")
                patch_vllm_attention_layer(module, config)
                
        self._attention_patched = True
        
    def _is_attention_layer(self, name: str) -> bool:
        """Check if module is an attention layer."""
        attention_indicators = ['attn', 'attention', 'self_attention']
        return any(indicator in name.lower() for indicator in attention_indicators)
        
    @property
    def tokenformer_manager(self):
        """Get the tokenformer manager."""
        return self._tokenformer_manager
        
    def __getattr__(self, name):
        """Delegate to the wrapped vLLM model for all other attributes."""
        return getattr(self.vllm_model, name)


class AdapterManager:
    """
    Manager that coordinates ScalarLM adapter functionality.
    Handles initialization and provides a simplified interface to the adapter system.
    """
    
    def __init__(self):
        self._adapter: Optional[ScalarLMAdapter] = None
        self._initialized = False
        
    def initialize(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_provider: Optional[ConfigProvider] = None
    ):
        """Initialize ScalarLM integration."""
        if self._initialized:
            logger.info("ScalarLM integration already initialized")
            return
            
        logger.info("Initializing ScalarLM integration with clean architecture")
        
        # Create config provider
        if config_provider is None:
            config_provider = ScalarLMConfigProvider(config)

        # Create adapter
        self._adapter = ScalarLMAdapter(config_provider=config_provider)
        self._initialized = True

        logger.info("ScalarLM integration initialized successfully")
        
    def enhance_model(self, model, device: torch.device, model_id: Optional[str] = None):
        """Enhance a vLLM model with ScalarLM functionality."""
        if not self._initialized:
            raise RuntimeError("ScalarLM integration not initialized. Call initialize() first.")
            
        return self._adapter.enhance_vllm_model(model, device, model_id)
        
    def create_tokenformer_manager(self, model, device: torch.device):
        """Create a tokenformer manager."""
        if not self._initialized:
            raise RuntimeError("ScalarLM integration not initialized. Call initialize() first.")
            
        return self._adapter.create_tokenformer_manager(model, device)
        
    def is_initialized(self) -> bool:
        """Check if integration is initialized."""
        return self._initialized


# Global manager instance
_manager = AdapterManager()


def init_adapters(
    config: Optional[Dict[str, Any]] = None,
    config_provider: Optional[ConfigProvider] = None
):
    """Initialize the ScalarLM adapter system for enhancing vLLM models."""
    _manager.initialize(config, config_provider)




def enhance_vllm_model(model, device: torch.device, model_id: Optional[str] = None):
    """Enhance a vLLM model with ScalarLM functionality."""
    if not _manager.is_initialized():
        init_adapters()
    return _manager.enhance_model(model, device, model_id)


def create_tokenformer_manager(model, device: torch.device):
    """Create a tokenformer manager for a model."""
    if not _manager.is_initialized():
        init_adapters()
    return _manager.create_tokenformer_manager(model, device)


@contextmanager
def enhanced_vllm(model, device: torch.device, **config):
    """Context manager for using ScalarLM-enhanced vLLM models."""
    # Initialize with config
    init_adapters(config)
    
    # Enhance model
    enhanced_model = enhance_vllm_model(model, device)
    
    try:
        yield enhanced_model
    finally:
        # Cleanup if needed
        pass

# Backward compatibility alias
scalarlm_enhanced_vllm = enhanced_vllm


# Internal access (not part of public API)
def _get_manager() -> AdapterManager:
    """Get the global adapter manager (internal use only)."""
    return _manager
