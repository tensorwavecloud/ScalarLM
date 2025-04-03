import argparse
import time
import torch
from gpu_aware_mpi import allgather, allreduce, reduce_scatter, barrier, get_rank, get_size, finalize_mpi

def create_buffer(arch, size, rank):
    if arch == 'cuda':
        return torch.ones(size, dtype=torch.float32, device='cuda')
    elif arch == 'rocm':
        return torch.ones(size, dtype=torch.float32, device='cuda:' + str(rank))  # ROCm uses the same device string
    else:
        return torch.ones(size, dtype=torch.float32, device='cpu')

def benchmark_collective(collective_fn, send_size, recv_size, expected_value, num_iters=100, warmup=10):
    size = get_size()
    rank = get_rank()
    sendbuf = create_buffer(args.arch, send_size, rank).contiguous()
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

    # Calculate bandwidth (we use float32 for ReduceScatter and AllReduce internally)
    datatype_bytes = 4
    total_data = send_size * datatype_bytes * 2 * size  # 4 bytes per float32, 2 for send/recv
    bandwidth = (total_data / dt) / 1e9  # GB/s
    return bandwidth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['cuda', 'rocm', 'cpu'], required=True)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32', help="Data type for buffers")
    args = parser.parse_args()

    args.dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    data_size = 262144 * 64 
    rank = get_rank()
    size = get_size()

    collectives = {
         'AllGather': (lambda sbuf, rbuf: allgather(sbuf, rbuf), data_size, data_size * size, 1.0),
         'ReduceScatter': (lambda sbuf, rbuf: reduce_scatter(sbuf, rbuf), data_size, data_size // size, size * 1.0),
         'AllReduce': (lambda sbuf, rbuf: allreduce(sbuf), data_size, data_size, size * 1.0),
     }

    results = {}
    
    for name, info in collectives.items():
         bw = benchmark_collective(info[0], info[1], info[2], info[3])
         if rank == 0:
             results[name] = bw

    if rank == 0:
        print("\nBenchmark Results (GB/s):")
        for name, bw in results.items():
            bw_scientific = '{:.2e}'.format(bw)
            print(f"{name}: {bw_scientific}")
    
    finalize_mpi()


# For CUDA GPUs
# mpirun --allow-run-as-root --oversubscribe -np 4 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch cuda

# For ROCm GPUs
# mpirun --allow-run-as-root -np 4 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch rocm

# For CPU
# mpirun --allow-run-as-root --oversubscribe -np 2 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch cpu
