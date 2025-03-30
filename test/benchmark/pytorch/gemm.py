import torch
import time
from tqdm import tqdm
import json

import logging

logger = logging.getLogger(__name__)


def main():
    benchmark_gemm()


def benchmark_gemm():
    logger.info("Running GEMM benchmark")
    results = run_gemm_benchmark()

    save_results(results)


def run_gemm_benchmark():

    warmup()

    results = {}

    for size in tqdm(gemm_sizes):
        results[str(size)] = run_gemm(size)

    return results


def warmup():
    run_gemm(gemm_sizes[0])


def run_gemm(size):
    m, n, k = size
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    c = torch.randn(m, n, dtype=torch.float16)

    # run at least 1 second
    start_time = time.time()
    end_time = start_time + 1

    barrier()

    start = get_event()
    end = get_event()

    start.record()
    iterations = 0
    while time.time() < end_time:
        c = torch.mm(a, b)
        iterations += 1
    end.record()

    barrier()

    seconds = start.elapsed_time(end) / 1000 / iterations

    return {
        "size": size,
        "time": seconds,
        "GFLOP/s": 2 * m * n * k / seconds / 1e9,
    }


class CPUEvent:
    def __init__(self):
        self.time = 0

    def record(self):
        self.time = time.time()

    def elapsed_time(self, other):
        return (other.time - self.time) * 1000


def get_event():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    else:
        return CPUEvent()


def barrier():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass


def select_appropriate_size_for_this_machine():
    # Try a small GEMM, and time it
    # If it runs too fast, select a bigger model
    # If it runs too slow, select a smaller model

    tiny_gemm = (256, 256, 256)

    metrics = run_gemm(tiny_gemm)

    # Get the total number of flops in each model

    def get_flops(size):
        m, n, k = size
        return 2 * m * n * k

    tiny_flops = get_flops(tiny_gemm)

    # get the flops for each llama model
    llama_100m_flops = sum([get_flops(size) for size in llama_100m_sizes])
    llama_1b_flops = sum([get_flops(size) for size in llama_1b_sizes])
    llama_8b_flops = sum([get_flops(size) for size in llama_8b_sizes])

    # Get the time it took to run the tiny gemm
    tiny_time = metrics["time"]

    # Get the time it would take to run each llama model
    llama_100m_time = llama_100m_flops / tiny_flops * tiny_time
    llama_1b_time = llama_1b_flops / tiny_flops * tiny_time
    llama_8b_time = llama_8b_flops / tiny_flops * tiny_time

    # Select the smallest model that will take at most 10 seconds to run
    if llama_100m_time < 10:
        return llama_100m_sizes
    elif llama_1b_time < 10:
        return llama_1b_sizes
    elif llama_8b_time < 10:
        return llama_8b_sizes
    else:
        return [tiny_gemm]


def save_results(results):
    # Save results to a json file
    path = "/app/cray/data/benchmark_gemm.json"

    with open(path, "w") as f:
        json.dump(results, f)

# Common sizes used in LLMs

llama_100m_sizes = [
    (256, 256, 2048),
    (16384, 256, 2048),
    (2048, 256, 1024),
]

llama_1b_sizes = [
    (2048, 2048, 2048),
    (128256, 2048, 2048),
    (2048, 2048, 8192),
]


llama_8b_sizes = [
    (4096, 4096, 2048),
    (128256, 4096, 2048),
    (2048, 4096, 14336),
]

gemm_sizes = select_appropriate_size_for_this_machine()


if __name__ == "__main__":
    main()
