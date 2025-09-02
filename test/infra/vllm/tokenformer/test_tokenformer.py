import pytest
import torch
from pathlib import Path

from cray_infra.adapters.model.tokenformer import TokenformerModel

@pytest.fixture
def mock_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def mock_tokenformer_tensor(mock_model_dir):
    tensor_path = mock_model_dir / "model.pt"
    mock_tensors = {
        "tokenformer.weight": torch.rand(10, 10),
        "lm_head.weight": torch.rand(10),
        "other_module.weight": torch.rand(5, 5)  # This should not be included
    }
    # Save as .pt file (PyTorch format) instead of safetensors
    torch.save(mock_tensors, tensor_path)
    return tensor_path

def test_from_local_checkpoint(mock_model_dir, mock_tokenformer_tensor):
    device = torch.device("cpu")  # Use CPU device for testing
    model = TokenformerModel.from_local_checkpoint(str(mock_model_dir), device)
    
    assert isinstance(model, TokenformerModel)
    assert len(model.tokenformers) == 2
    assert "tokenformer.weight" in model.tokenformers
    assert "lm_head.weight" in model.tokenformers
    assert "other_module.weight" not in model.tokenformers
    assert isinstance(model.tokenformers["tokenformer.weight"], torch.Tensor)
    assert isinstance(model.tokenformers["lm_head.weight"], torch.Tensor)

def test_from_local_checkpoint_file_not_found(mock_model_dir):
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        TokenformerModel.from_local_checkpoint(str(mock_model_dir), device)

def test_from_local_checkpoint_invalid_dir():
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        TokenformerModel.from_local_checkpoint("non/existent/path", device)

# Allow running this test file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
