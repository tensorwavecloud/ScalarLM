import pytest
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from ml.tokenformer.llama_tokenformer_model import create_llama_tokenformer_model


@pytest.fixture(scope="module")
def model_setup():
    model_name = "masint/tiny-random-llama"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model1 = LlamaForCausalLM.from_pretrained(model_name).to(device)
    model2 = create_llama_tokenformer_model(model1).to(device)
    return model1, model2, tokenizer, device


def compare_model_outputs(model1, model2, tokenizer, input_text, device):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output1 = model1(**inputs).logits
        output2 = model2(**inputs).logits
    is_close = torch.allclose(output1, output2, rtol=1e-4, atol=1e-4)
    max_diff = torch.max(torch.abs(output1 - output2)).item()
    return is_close, max_diff


@pytest.mark.parametrize(
    "input_text",
    [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "In a world of artificial intelligence, creativity still matters.",
    ],
)
def test_tokenformer_model_equivalence(model_setup, input_text):
    model1, model2, tokenizer, device = model_setup
    is_close, max_diff = compare_model_outputs(
        model1, model2, tokenizer, input_text, device
    )

    assert is_close, f"Outputs are not close for input: {input_text}"
    assert (
        max_diff < 1e-3
    ), f"Maximum difference {max_diff} is too large for input: {input_text}"
