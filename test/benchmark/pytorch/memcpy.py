import torch
import json
import time
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

# List of memcpy sizes, in bytes, should be multiples of the page size
memcpy_sizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def main():
    benchmark_memcpy()


def benchmark_memcpy():
    logger.info("Running memcpy benchmark")
    results = run_memcpy_benchmark()

    save_results(results)


def run_memcpy_benchmark():

    warmup()

    results = {}

    for size in tqdm(memcpy_sizes):
        results[size] = run_memcpy(size)

    return results


def warmup():
    run_memcpy(4096)


def run_memcpy(size):
    a = torch.randn(size)
    b = torch.randn(size)

    # copy at least 16MB
    total_data_copied = 16 * 1024 * 1024

    iterations = total_data_copied // size

    barrier()

    start = get_event()
    end = get_event()

    start.record()
    for _ in range(iterations):
        b.copy_(a)
    end.record()

    barrier()
    time = start.elapsed_time(end) * 1e-3 / iterations

    return {
        "size": size,
        "time": time,
        "bandwidth": size / time,
        "GB/s": size / time / 1e9,
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
    path = "/app/cray/data/benchmark_memcpy.json"

    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
