import pytest
import torch
from pathlib import Path
from safetensors.torch import save_file

from infra.cray_infra.vllm.tokenformer.tokenformer_model_manager import TokenformerModel  # Replace 'your_module' with the actual module name

@pytest.fixture
def mock_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def mock_tokenformer_tensor(mock_model_dir):
    tensor_path = mock_model_dir / "model.safetensors"
    mock_tensors = {
        "tokenformer.weight": torch.rand(10, 10),
        "lm_head.weight": torch.rand(10),
        "other_module.weight": torch.rand(5, 5)  # This should not be included
    }
    save_file(mock_tensors, tensor_path)
    return tensor_path

def test_from_local_checkpoint(mock_model_dir, mock_tokenformer_tensor):
    model = TokenformerModel.from_local_checkpoint(str(mock_model_dir))
    
    assert isinstance(model, TokenformerModel)
    assert len(model.tokenformers) == 2
    assert "tokenformer.weight" in model.tokenformers
    assert "lm_head.weight" in model.tokenformers
    assert "other_module.weight" not in model.tokenformers
    assert isinstance(model.tokenformers["tokenformer.weight"], torch.Tensor)
    assert isinstance(model.tokenformers["lm_head.weight"], torch.Tensor)

def test_from_local_checkpoint_file_not_found(mock_model_dir):
    with pytest.raises(FileNotFoundError):
        TokenformerModel.from_local_checkpoint(str(mock_model_dir))

def test_from_local_checkpoint_invalid_dir():
    with pytest.raises(FileNotFoundError):
        TokenformerModel.from_local_checkpoint("non/existent/path")
