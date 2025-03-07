import pytest
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from ml.tokenformer.llama_tokenformer_model import create_llama_tokenformer_model

@pytest.fixture
def model_setup():
    model_name = "masint/tiny-random-llama"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    return model

def test_create_llama_tokenformer_model(model_setup):
    model = model_setup
    # lm_head is trained by default
    result = create_llama_tokenformer_model(model, "cpu")

    # Check requires_grad is set correctly 
    for name, param in result.named_parameters():
        if any(module_name in name for module_name in ["tokenformer", "lm_head"]):
            assert param.requires_grad
        else:
            assert not param.requires_grad

def test_create_llama_tokenformer_model_no_lm_head(model_setup):
    model = model_setup
    # lm_head should not be trained
    result = create_llama_tokenformer_model(model=model, device="cpu", train_lm_head=False)

    # Check requires_grad is set correctly 
    for name, param in result.named_parameters():
        if any(module_name in name for module_name in ["tokenformer"]):
            assert param.requires_grad
        else:
            assert not param.requires_grad
