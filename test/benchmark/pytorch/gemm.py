import torch
import time
from tqdm import tqdm
import json

import logging

logger = logging.getLogger(__name__)

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

gemm_sizes = llama_100m_sizes


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
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    c = torch.randn(m, n, dtype=torch.float32)

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


def save_results(results):
    # Save results to a json file
    path = "/app/cray/data/benchmark_gemm.json"

    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
