import pytest
import torch
import torch.nn as nn
from ml.tokenformer.tokenformer_surgeon import TokenformerMLPAdapter
from ml.tokenformer.transformers_tokenformer import TransformersTokenformerSurgeon, TransformersTokenformerAttentionAdapter

class MockAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        return torch.randn_like(query)

class MockMLP(nn.Module):
    def forward(self, x):
        return x

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MockAttention(64)
        self.mlp = MockMLP()
        self.config = type('Config', (), {'hidden_size': 768})()

@pytest.fixture
def mock_model():
    return MockModel()

def test_transformers_tokenformer_surgeon_initialization(mock_model):
    surgeon = TransformersTokenformerSurgeon(mock_model, "cpu")
    assert surgeon.model == mock_model

def test_is_attn_layer():
    surgeon = TransformersTokenformerSurgeon(None, "cpu")
    assert surgeon._is_attn_layer("model.layer.0.attn")
    assert not surgeon._is_attn_layer("model.layer.0.mlp")

def test_is_mlp_layer():
    surgeon = TransformersTokenformerSurgeon(None, "cpu")
    assert surgeon._is_mlp_layer("model.layer.0.mlp")
    assert not surgeon._is_mlp_layer("model.layer.0.attn")

def test_recursive_setattr(mock_model):
    surgeon = TransformersTokenformerSurgeon(mock_model, "cpu")
    new_mlp = TokenformerMLPAdapter(mock_model.mlp, 768, "cpu")
    surgeon._recursive_setattr(mock_model, "mlp", new_mlp)
    assert mock_model.mlp == new_mlp

def test_update_attn(mock_model):
    surgeon = TransformersTokenformerSurgeon(mock_model, "cpu")
    assert not isinstance(mock_model.attn, TransformersTokenformerAttentionAdapter)
    surgeon.update_attn("attn", mock_model.attn)
    assert isinstance(mock_model.attn, TransformersTokenformerAttentionAdapter)

def test_insert_adapter_modules(mock_model):
    surgeon = TransformersTokenformerSurgeon(mock_model, "cpu")
    assert not isinstance(surgeon.model.attn, TransformersTokenformerAttentionAdapter)
    updated_model = surgeon.insert_adapter_modules()
    assert isinstance(updated_model.attn, TransformersTokenformerAttentionAdapter)

def test_tokenformer_mlp_adapter():
    mock_layer = MockMLP()
    adapter = TokenformerMLPAdapter(mock_layer, 768, "cpu")
    input_tensor = torch.randn(1, 10, 768)
    output = adapter(input_tensor)
    assert output.shape == input_tensor.shape


