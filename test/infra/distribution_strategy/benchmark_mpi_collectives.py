import argparse
import time
import torch
from gpu_aware_mpi import allgather, reduce_scatter, send, recv, barrier, get_rank, get_size, mpi_finalize

def create_buffer(arch_type, size, rank):
    if arch_type == 'cuda':
        return torch.ones(size, dtype=torch.float32, device='cuda')
    elif arch_type == 'rocm':
        return torch.ones(size, dtype=torch.float32, device='cuda:' + str(rank))  # ROCm uses the same device string
    else:
        return torch.ones(size, dtype=torch.float32, device='cpu')

def benchmark_collective(collective_fn, send_size, recv_size, expected_value, num_iters=100, warmup=10):
    size = get_size()
    rank = get_rank()
    sendbuf = create_buffer(args.arch_type, send_size, rank).contiguous()
    # Create send/recv buffers
    recvbuf = torch.empty(recv_size, dtype=torch.float32, device=sendbuf.device).contiguous()

    # Warmup iterations
    for _ in range(warmup):
        collective_fn(sendbuf, recvbuf)

    # Timing the collective operation
    barrier()
    t0 = time.time()
    for _ in range(num_iters):
        collective_fn(sendbuf, recvbuf)
    barrier()
    dt = time.time() - t0

    # Verify correctness
    assert torch.allclose(recvbuf, torch.full_like(recvbuf, expected_value), atol=1e-6), "Verification failed"

    # Calculate bandwidth (assumes float32)
    total_data = send_size * 4 * 2 * size  # 4 bytes per float32, 2 for send/recv
    bandwidth = (total_data / dt) / 1e9  # GB/s
    return bandwidth

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
    
    print(f"Rank {rank} of {size} running with data size {data_size}")

    bandwidth = benchmark_send_recv(data_size)

    if rank == 0:
        print(f"Send/Recv (GB/s): {bandwidth:.6}")
        
    mpi_finalize()

    # collectives = {
    #     'AllGather': (lambda sbuf, rbuf: allgather(sbuf, rbuf), data_size, data_size * size, 1.0),
    #     'ReduceScatter': (lambda sbuf, rbuf: reduce_scatter(sbuf, rbuf), data_size, data_size // size, size * 1.0),
    # }

    # results = {}
    #
    # for name, info in collectives.items():
    #     bw = benchmark_collective(info[0], info[1], info[2], info[3])
    #     if rank == 0:
    #         results[name] = bw

    # if rank == 0:
    #     print("\nBenchmark Results (GB/s):")
    #     for name, bw in results.items():
    #         bw_scientific = '{:.2e}'.format(bw)
    #         print(f"{name}: {bw_scientific}")


# For CUDA GPUs
# mpirun --allow-run-as-root --oversubscribe -np 4 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type cuda

# For ROCm GPUs
# mpirun --allow-run-as-root -np 4 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type rocm

# For CPU
# mpirun --allow-run-as-root --oversubscribe -np 2 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type cpu
