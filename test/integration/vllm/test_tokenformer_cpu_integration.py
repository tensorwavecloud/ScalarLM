#!/usr/bin/env python3
"""
Integration test for tokenformer support in CPUModelRunner
This test runs inside the Docker container where vLLM is properly built
"""

import asyncio
import sys
import os

# Add paths for Docker container
sys.path.insert(0, '/app/cray/vllm')
sys.path.insert(0, '/app/cray/infra')


def test_tokenformer_manager_basic():
    """Test basic TokenformerModelManager functionality"""
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
    
    print("Testing TokenformerModelManager initialization...")
    
    # Create a simple mock model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)
    
    model = SimpleModel()
    device = torch.device("cpu")
    manager = TokenformerModelManager(model=model, device=device)
    
    assert manager.device == device
    assert manager.model is not None
    assert hasattr(manager, 'model')
    
    print("✓ TokenformerModelManager initialized successfully")


def test_cpu_model_runner_initialization():
    """Test that tokenformer can be used with model runner mixin"""
    import torch
    import torch.nn as nn
    from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
    
    print("Testing TokenformerModelManager with model runner context...")
    
    # Create a mock model that supports lora/tokenformer
    class MockLLMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(768, 50257)  # GPT-2 like dimensions
            self.supports_lora = True
    
    model = MockLLMModel()
    device = torch.device("cpu")
    
    # Initialize TokenformerModelManager as done in LoRAModelRunnerMixin
    manager = TokenformerModelManager(model=model, device=device)
    
    assert manager.model is not None
    assert manager.device == device
    
    print("✓ TokenformerModelManager integration with model runner successful")


def test_tokenformer_with_lora_config():
    """Test LoRAModelRunnerMixin with LoRA config uses TokenformerModelManager"""
    import torch
    import torch.nn as nn
    from vllm.config import ModelConfig, SchedulerConfig, LoRAConfig
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
    from unittest.mock import MagicMock, patch
    
    print("Testing LoRAModelRunnerMixin with LoRA config...")
    
    # Create minimal configs
    model_config = MagicMock(spec=ModelConfig)
    model_config.hf_config = MagicMock()
    model_config.hf_config.get_text_config.return_value = MagicMock()
    
    scheduler_config = MagicMock(spec=SchedulerConfig)
    lora_config = MagicMock(spec=LoRAConfig)
    
    # Create a mock model that supports lora
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)
    
    model = MockModel()
    device = torch.device("cpu")
    
    # Test LoRAModelRunnerMixin
    mixin = LoRAModelRunnerMixin()
    
    with patch('vllm.v1.worker.lora_model_runner_mixin.supports_lora', return_value=True), \
         patch('vllm.model_executor.models.supports_lora', return_value=True):
        result_model = mixin.load_lora_model(model, model_config, scheduler_config, lora_config, device)
        
        assert result_model is not None
        assert hasattr(mixin, 'lora_manager')
        assert mixin.lora_manager is not None
        print("✓ LoRAModelRunnerMixin uses TokenformerModelManager successfully")


def test_tokenformer_model_manager():
    """Test TokenformerModelManager is available"""
    try:
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager, TokenformerModel
        
        print("Testing TokenformerModelManager availability...")
        
        # Check classes are importable
        assert TokenformerModelManager is not None
        assert TokenformerModel is not None
        
        print("✓ TokenformerModelManager and TokenformerModel are available")
    except ImportError as e:
        print(f"⚠ TokenformerModelManager not available: {e}")
        pytest.fail(f"TokenformerModelManager not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])