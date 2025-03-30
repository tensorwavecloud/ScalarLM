import torch
import time
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

import logging

logger = logging.getLogger(__name__)


def main():
    benchmark_backward()


def benchmark_backward():
    logger.info("Running Backward benchmark")
    results = run_backward_benchmark()

    save_results(results)


def run_backward_benchmark():

    models = select_appropriate_models_for_this_machine()

    warmup(
        model_name=models[0][0],
        batch_size=models[0][1],
        input_tokens=models[0][2],
    )

    results = {}

    for model_name, batch_size, input_tokens in tqdm(models):
        result = run_backward(
            model_name=model_name,
            batch_size=batch_size,
            input_tokens=input_tokens,
        )

        if model_name not in results:
            results[model_name] = result
        else:
            results[model_name].update(result)

    return results


def warmup(model_name, batch_size, input_tokens):
    run_backward(
        model_name=model_name,
        batch_size=batch_size,
        input_tokens=input_tokens,
    )


def run_backward(model_name, batch_size, input_tokens):

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Input tokens are randomly generated ints between 0 and the model's vocab size
    input_ids = torch.randint(
        low=0, high=model.config.vocab_size, size=(batch_size, input_tokens)
    )

    # Run the backward pass
    start = time.time()
    outputs = model.forward(input_ids, labels=input_ids)
    outputs.loss.backward()
    end = time.time()

    flop_count = calculate_flop_count(model, batch_size, input_tokens)

    return {
        input_tokens: {
            batch_size: {
                "time": end - start,
                "GFLOP/s": flop_count / (end - start) / 1e9,
                "flop_count": flop_count,
                "batch_size": batch_size,
                "input_tokens": input_tokens,
            }
        }
    }


def calculate_flop_count(model, batch_size, input_tokens):
    # parameter count
    param_count = sum(p.numel() for p in model.parameters())

    # GEMM flops
    gemm_flops = 3 * 2 * batch_size * (input_tokens) * param_count

    return gemm_flops


def save_results(results):
    # Save results to a json file
    path = "/app/cray/data/benchmark_backward.json"

    with open(path, "w") as f:
        json.dump(results, f)


def select_appropriate_models_for_this_machine():
    # Run a GEMM operation to determine the peak GEMM flops of the machine
    # This will be used to determine which models to run the benchmark on

    gemm_flops = calculate_gemm_flops()

    # Machines with over 1TFLOP/s peak GEMM performance can run the llama models
    # Over 10 TFLOP/s can run the 70B model

    if gemm_flops > 10e12:
        return benchmark_model_list
    elif gemm_flops > 1e12:
        return benchmark_model_list[:-1]
    else:
        return benchmark_model_list[:1]


def calculate_gemm_flops():
    m, n, k = 256, 256, 256

    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)

    start = time.time()
    c = torch.matmul(a, b)
    end = time.time()

    return 2 * m * n * k / (end - start)


benchmark_model_list = [
    ["masint/tiny-random-llama", 1, 128],
    ["meta-llama/Llama-3.2-1B-Instruct", 1, 128],
    ["meta-llama/Llama-3.1-8B-Instruct", 1, 128],
    ["meta-llama/Llama-3.3-70B-Instruct", 1, 128],
]

