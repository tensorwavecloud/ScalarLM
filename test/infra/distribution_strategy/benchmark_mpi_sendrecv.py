import argparse
import time
import torch
from gpu_aware_mpi import send, recv, barrier, get_rank, get_size, finalize_mpi

def create_buffer(arch_type, size, rank):
    if arch_type == 'cuda':
        return torch.ones(size, dtype=torch.float32, device='cuda')
    elif arch_type == 'rocm':
        return torch.ones(size, dtype=torch.float32, device='cuda:' + str(rank))  # ROCm uses the same device string
    else:
        return torch.ones(size, dtype=torch.float32, device='cpu')

def send_recv(sendbuf, i):
    sender = i % 2
    rank = get_rank()
    barrier()
    if rank == sender:
        send(sendbuf, (rank + 1) % 2)
    else:
        recv(sendbuf, (rank + 1) % 2)
    barrier()

def benchmark_send_recv(data_size, num_iters=100, warmup=10):
    rank = get_rank()
    sendbuf = create_buffer(args.arch_type, data_size, rank).contiguous()

    if rank == 1:
        torch.zero_(sendbuf)

    for i in range(0, warmup * 2, 2):
        send_recv(sendbuf, i)

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
    return bandwidth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_type', choices=['cuda', 'rocm', 'cpu'], required=True)
    args = parser.parse_args()

    data_size = 262144 * 64 
    rank = get_rank()
    size = get_size()
    
    assert size == 2, "This test only works with two ranks."
    
    print(f"Rank {rank} of {size} running with data size {data_size}")

    bandwidth = benchmark_send_recv(data_size)

    if rank == 0:
        print(f"Send/Recv (GB/s): {bandwidth:.6}")
        
    finalize_mpi()