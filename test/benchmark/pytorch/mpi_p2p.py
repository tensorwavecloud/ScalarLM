import torch
import json
import time
import os

from gpu_aware_mpi import send, recv, barrier, get_rank, get_size, finalize_mpi

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

# List of memcpy sizes, in bytes, should be multiples of the page size
memcpy_sizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

def main():
    benchmark_mpi_p2p()

def benchmark_mpi_p2p():
    warmup()

    results = {}

    for size in tqdm(memcpy_sizes):
        results[size] = benchmark_send_recv(size)

    save_results(results)

def send_recv(sendbuf, i):
    sender = i % 2
    rank = get_rank()
    barrier()
    if rank == sender:
        send(sendbuf, (rank + 1) % 2)
    else:
        recv(sendbuf, (rank + 1) % 2)
    barrier()

def benchmark_send_recv(data_size):
    rank = get_rank()
    sendbuf = create_buffer(data_size, rank).contiguous()

    if rank == 1:
        torch.zero_(sendbuf)

    # Timing the collective operation
    barrier()
    t0 = time.time()
    for i in range(0, num_iters * 2, 2):
        send_recv(sendbuf, i)
    barrier()
    dt = time.time() - t0

    # Verify correctness
    assert torch.allclose(sendbuf, torch.full_like(sendbuf, 1), atol=1e-6), "Verification failed"

    total_data = data_size * 4 * num_iters
    bandwidth = (total_data / dt) / 1e9  # GB/s
    return {
        "size": data_size,
        "time": dt,
        "bandwidth": total_data / dt,
        'GB/s': bandwidth
    }

def create_buffer(size, rank):
    device = torch.device(rank)

    return torch.full((size // 4,), rank + 1, dtype=torch.float32, device=device)


def launch_mpi_p2p():
    this_script_path = os.path.dirname(os.path.realpath(__file__))

    srun_command = f"srun -n 2 python {this_script_path}/mpi_p2p.py"

    logger.info(f"Running command: {srun_command}")

    os.system(srun_command)


if __name__ == '__main__':
    main()

