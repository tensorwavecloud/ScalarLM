"""
ScalarLM Adapter Commons - Extracted from vLLM to maintain clean separation.
This replaces vllm.adapter_commons completely.
"""

from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AdapterModel(ABC):
    """Base class for adapter models (replaces vllm.adapter_commons.models.AdapterModel)."""
    
    def __init__(self, adapter_id: str):
        self.id = adapter_id


class AdapterModelManager(ABC):
    """Base class for adapter model managers (replaces vllm.adapter_commons.models.AdapterModelManager)."""
    
    @abstractmethod
    def add_adapter(self, adapter: AdapterModel) -> bool:
        """Add an adapter to the manager."""
        pass
        
    @abstractmethod
    def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter from the manager."""
        pass
        
    @abstractmethod
    def activate_adapter(self, adapter_id: str) -> bool:
        """Activate a specific adapter."""
        pass
        
    @abstractmethod
    def deactivate_adapter(self, adapter_id: str) -> bool:
        """Deactivate a specific adapter.""" 
        pass


# Utility functions (replaces vllm.adapter_commons.utils)
def get_adapter(adapter_id: str, adapters: Dict[str, Any]) -> Optional[Any]:
    """Get an adapter by ID."""
    return adapters.get(adapter_id)


def list_adapters(adapters: Dict[str, Any]) -> Dict[str, Any]:
    """List all adapters."""
    return adapters.copy()


def remove_adapter(adapter_id: str, adapters: Dict[str, Any], remove_fn) -> bool:
    """Remove an adapter using the provided removal function."""
    if adapter_id in adapters:
        remove_fn(adapter_id)
        return True
    return False


def deactivate_adapter(adapter_id: str, active_adapter_ref: list) -> bool:
    """Deactivate an adapter."""
    if active_adapter_ref and active_adapter_ref[0] == adapter_id:
        active_adapter_ref[0] = None
        return True
    return False


class ConfigProvider(Protocol):
    """Protocol for providing configuration to adapters without vLLM knowing about ScalarLM."""
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...


class DefaultConfigProvider:
    """Default configuration provider that uses ScalarLM's config system."""
    
    def __init__(self):
        self._config_cache = {}
        self._loaded = False
        
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value, loading ScalarLM config as needed."""
        if not self._loaded:
            self._load_scalarlm_config()
            
        return self._config_cache.get(key, default)
        
    def _load_scalarlm_config(self):
        """Load ScalarLM configuration without vLLM depending on it."""
        try:
            # Import here to avoid vLLM depending on ScalarLM
            from cray_infra.util.get_config import get_config
            config = get_config()
            
            # Extract commonly used config values
            self._config_cache = {
                "tokenformer_cache_capacity": getattr(config, 'tokenformer_cache_capacity', 4),
                "enable_tokenformer": getattr(config, 'enable_tokenformer', True),
                # Add other config values as needed
            }
            self._loaded = True
            
        except ImportError:
            logger.warning("ScalarLM config not available, using defaults")
            self._config_cache = {
                "tokenformer_cache_capacity": 4,
                "enable_tokenformer": True,
            }
            self._loaded = True


# Global config provider
_config_provider: Optional[ConfigProvider] = None


def set_config_provider(provider: ConfigProvider):
    """Set the global configuration provider."""
    global _config_provider
    _config_provider = provider


def get_config_provider() -> ConfigProvider:
    """Get the global configuration provider."""
    global _config_provider
    if _config_provider is None:
        _config_provider = DefaultConfigProvider()
    return _config_provider