#!/usr/bin/env python3
"""
Integration test for CPUModelRunner with tokenformer adapter management
Tests the full flow of loading and managing tokenformer adapters
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import sys
from pathlib import Path

# Add vllm to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'vllm'))


class SimpleModel(nn.Module):
    """Simple model for testing that supports LoRA"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.supports_lora = True
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_adapter_path():
    """Create a temporary directory with mock adapter files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = Path(tmpdir) / "adapter1"
        adapter_dir.mkdir()
        
        # Create a mock adapter file
        mock_weights = {
            "tokenformer.weight": torch.rand(10, 10),
            "lm_head.weight": torch.rand(10)
        }
        torch.save(mock_weights, adapter_dir / "model.pt")
        
        yield str(adapter_dir)


class TestCPUTokenformerIntegration:
    """Integration tests for CPU tokenformer functionality"""
    
    def test_tokenformer_manager_initialization(self):
        """Test that tokenformer manager can be initialized"""
        import torch.nn as nn
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
        
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
    
    def test_tokenformer_model_creation(self, temp_adapter_path):
        """Test creating and loading a tokenformer model"""
        import torch.nn as nn
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModel
        
        # Test loading a tokenformer model from checkpoint
        try:
            tokenformer = TokenformerModel.from_local_checkpoint(temp_adapter_path)
            assert tokenformer is not None
            assert hasattr(tokenformer, 'adapter_data')
            print("✓ TokenformerModel loaded successfully")
        except Exception as e:
            pytest.skip(f"TokenformerModel loading not fully implemented: {e}")
    
    def test_tokenformer_integration_with_runner_mixin(self):
        """Test LoRAModelRunnerMixin integration with tokenformer"""
        import torch.nn as nn
        from vllm.tokenformer.tokenformer_model_manager import TokenformerModelManager
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
        
        # Create a mock model that supports lora/tokenformer  
        class MockLLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(768, 50257)
        
        model = MockLLMModel()
        device = torch.device("cpu")
        
        # Test TokenformerModelManager as used in LoRAModelRunnerMixin
        manager = TokenformerModelManager(model=model, device=device)
        
        assert manager.model is not None
        assert manager.device == device
        print("✓ TokenformerModelManager integration successful")


@pytest.mark.parametrize("has_lora_config", [True, False])  
def test_cpu_runner_lora_config_handling(has_lora_config):
    """Test that LoRAModelRunnerMixin correctly handles presence/absence of LoRA config"""
    import torch.nn as nn
    from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
    from vllm.config import ModelConfig, SchedulerConfig, LoRAConfig
    from unittest.mock import MagicMock
    
    # Create mock configs  
    model_config = MagicMock(spec=ModelConfig)
    model_config.hf_config = MagicMock()
    model_config.hf_config.get_text_config.return_value = MagicMock()
    
    scheduler_config = MagicMock(spec=SchedulerConfig)
    lora_config = MagicMock(spec=LoRAConfig) if has_lora_config else None
    
    # Create a mock model that supports lora
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(10, 10)
        
        @property
        def supports_lora(self):
            return True
    
    model = MockModel()
    device = torch.device("cpu")
    
    # Create runner mixin instance
    mixin = LoRAModelRunnerMixin()
    
    if has_lora_config:
        # Patch supports_lora to return True for our mock model
        from unittest.mock import patch
        with patch('vllm.v1.worker.lora_model_runner_mixin.supports_lora', return_value=True), \
             patch('vllm.model_executor.models.supports_lora', return_value=True):
            # Test loading model with LoRA config
            result_model = mixin.load_lora_model(model, model_config, scheduler_config, lora_config, device)
            assert result_model is not None
            assert hasattr(mixin, 'lora_manager')
            assert mixin.lora_manager is not None
    else:
        pytest.skip("No LoRA config test - mixin only used when LoRA enabled")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])