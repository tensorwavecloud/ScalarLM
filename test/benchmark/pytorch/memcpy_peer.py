import torch
import json
import time

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

# List of memcpy sizes, in bytes, should be multiples of the page size
# Go up to the tensor size used in Llama 3 (4096 * 128256 * 4) = 2_101_346_304
memcpy_sizes = [2 ** i  for i in range(12, 64) if 2 ** i <= 2_101_346_304]


def main():
    benchmark_memcpy_peer()


def benchmark_memcpy_peer():
    logger.info("Running memcpy peer benchmark")
    results = run_memcpy_peer_benchmark()

    save_results(results)


def run_memcpy_peer_benchmark():

    warmup()

    results = {}

    for size in tqdm(memcpy_sizes):
        results[size] = run_memcpy_peer(size)

    return results


def warmup():
    run_memcpy_peer(4096)


def run_memcpy_peer(size):
    a = torch.zeros(size // 4, device=get_device(0), dtype=torch.float32)
    b = torch.zeros(size // 4, device=get_device(1), dtype=torch.float32)

    start = get_event()
    end = get_event()

    start_time = time.time()

    barrier()

    start.record()
    iterations = 0
    while time.time() - start_time < 1:
        b.copy_(a)
        iterations += 1
    end.record()

    barrier()
    total_time = start.elapsed_time(end) * 1e-3 / iterations

    return {
        "operational_intensity": 1 / 4,  # 1 FLOP per 4 bytes
        "flops/s" : size / 4 / total_time,
        "bytes": size,
        "time": total_time,
        "iterations": iterations,
        "bandwidth": size / total_time,
        "GB/s": size / total_time / 1e9,
    }


def get_device(index):
    if torch.cuda.is_available():
        max_devices = torch.cuda.device_count()
        return torch.device(f"cuda:{index % max_devices}")
    else:
        return torch.device("cpu")


class CPUEvent:
    def __init__(self):
        self.time = 0

    def record(self):
        self.time = time.time()

    def elapsed_time(self, other):
        return (other.time - self.time) * 1e3


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
    path = "/app/cray/data/benchmark_memcpy_peer.json"

    with open(path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
