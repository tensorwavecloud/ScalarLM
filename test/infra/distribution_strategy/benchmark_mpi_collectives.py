import argparse
import time
import torch
from mpi4py import MPI
from infra.cray_infra.training.distribution_strategy.mpi_utils import get_mpi_datatype

def create_buffer(arch_type, size):
    if arch_type == 'cuda':
        return torch.ones(size, dtype=torch.float32, device='cuda')
    elif arch_type == 'rocm':
        return torch.ones(size, dtype=torch.float32, device='cuda')  # ROCm uses the same device string
    else:
        return torch.ones(size, dtype=torch.float32, device='cpu')

def benchmark_collective(comm, collective_fn, send_size, recv_size, expected_value, num_iters=100, warmup=10):
    size = comm.Get_size()

    sendbuf = create_buffer(args.arch_type, send_size).contiguous()
    # Create send/recv buffers
    recvbuf = torch.empty(recv_size, dtype=torch.float32, device=sendbuf.device).contiguous()

    sendbuf_raw = MPI.memory.fromaddress(sendbuf.data_ptr(), sendbuf.nbytes)
    recvbuf_raw = MPI.memory.fromaddress(recvbuf.data_ptr(), recvbuf.nbytes)

    # Warmup iterations
    for _ in range(warmup):
        collective_fn([sendbuf_raw, get_mpi_datatype(sendbuf)], [recvbuf_raw, get_mpi_datatype(recvbuf)])

    # Timing the collective operation
    comm.Barrier()
    t0 = time.time()
    for _ in range(num_iters):
        collective_fn([sendbuf_raw, get_mpi_datatype(sendbuf)], [recvbuf_raw, get_mpi_datatype(recvbuf)])
    comm.Barrier()
    dt = time.time() - t0

    # Verify correctness
    assert torch.allclose(recvbuf, torch.full_like(recvbuf, expected_value), atol=1e-6), "Verification failed"

    # Calculate bandwidth (assumes float32)    
    total_data = send_size * 4 * 2 * (size - 1)  # 4 bytes per float32, 2 for send/recv
    bandwidth = (total_data / dt) / 1e9  # GB/s
    return bandwidth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch_type', choices=['cuda', 'rocm', 'cpu'], required=True)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Scale data size with number of processes (1MB per process)
    data_size = 262144 * size # 1MB per process (262144 floats = 1MB)

    collectives = {
        'AllGather': (lambda sbuf, rbuf: comm.Allgather(sbuf, rbuf), data_size, data_size * size, 1.0),
        'ReduceScatter': (lambda sbuf, rbuf: comm.Reduce_scatter(sbuf, rbuf, op=MPI.SUM), data_size, data_size // size, size * 1.0),
    }

    results = {}
    
    for name, info in collectives.items():
        bw = benchmark_collective(comm, info[0], info[1], info[2], info[3])
        if rank == 0:
            results[name] = bw

    if rank == 0:
        print("\nBenchmark Results (GB/s):")
        for name, bw in results.items():
            bw_scientific = '{:.2e}'.format(bw)
            print(f"{name}: {bw_scientific}")


# For CUDA GPUs
# mpirun --allow-run-as-root --oversubscribe -np 4 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type cuda

# For ROCm GPUs
# mpirun --allow-run-as-root --oversubscribe -np 2 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type rocm

# For CPU
# mpirun --allow-run-as-root --oversubscribe -np 2 python test/infra/distribution_strategy/benchmark_mpi_collectives.py --arch_type cpu

