import scalarlm

import logging

scalarlm.api_url = "http://localhost:8000"


def main():
    llm = scalarlm.SupermassiveIntelligence()

    gpu_count = 1
    node_count = 2

    status = llm.submit_slurm_job(
        code=get_code(), train_args={"gpus": gpu_count, "nodes": node_count},
    )

    print(status)


def get_code():
    return """
from gpu_aware_mpi import get_size, get_rank, send, recv, barrier
import torch
import time
import statistics
from cray_infra.training.training_job_context import training_job_context

def calculate_message_sizes():
    sizes = []

    # Small messages: 1B to 1KB (powers of 2)
    for i in range(0, 11):  # 1B to 1KB
        sizes.append(2 ** i)

    # Medium messages: 2KB to 1MB (powers of 2)
    for i in range(11, 21):  # 2KB to 1MB
        sizes.append(2 ** i)

    # Large messages: 2MB to 512MB (powers of 2)
    for i in range(21, 30):  # 2MB to 512MB
        sizes.append(2 ** i)

    # Very large messages: 1GB to 8GB (to saturate 800 Gbps)
    # At 800 Gbps = 100 GB/s, we need large messages to achieve peak bandwidth
    for i in range(30, 34):  # 1GB to 8GB
        sizes.append(2 ** i)

    return sizes

def send_recv_benchmark(size_bytes, warmup_iterations=5, timing_iterations=10):
    rank = get_rank()
    world_size = get_size()
    bisection_rank = world_size // 2

    if bisection_rank == 0:
        if rank == 0:
            print("Warning: Bisection rank is 0, skipping test to avoid deadlock.")
        return None

    # Calculate tensor size (assuming float32 = 4 bytes)
    tensor_elements = size_bytes // 4
    if tensor_elements == 0:
        tensor_elements = 1  # Minimum one element

    # Create tensor on GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_tensor = torch.ones(tensor_elements, dtype=torch.float32, device=device)

    # Determine communication partner (cross network bisection)
    neighbor = (rank + bisection_rank) % world_size
    is_sender = rank < bisection_rank

    # Warmup phase
    barrier()  # Sync before warmup
    for _ in range(warmup_iterations):
        if is_sender:
            send(my_tensor, neighbor)
        else:
            received_tensor = recv(torch.empty_like(my_tensor), neighbor)
    barrier()  # Sync after warmup

    # Timing phase
    timings = []

    for iteration in range(timing_iterations):
        barrier()  # Ensure all processes start together

        start_time = time.perf_counter()

        if is_sender:
            send(my_tensor, neighbor)
        else:
            received_tensor = recv(torch.empty_like(my_tensor), neighbor)

        end_time = time.perf_counter()
        timings.append(end_time - start_time)

        barrier()  # Sync after each iteration

    # Calculate statistics
    if len(timings) > 0:
        avg_time = statistics.mean(timings)
        min_time = min(timings)
        max_time = max(timings)
        std_time = statistics.stdev(timings) if len(timings) > 1 else 0.0

        # Calculate bandwidth (bytes per second)
        # Note: This is unidirectional bandwidth
        bandwidth_bps = size_bytes / avg_time
        bandwidth_gbps = bandwidth_bps / (1024**3)  # Convert to GB/s
        bandwidth_gbps_network = bandwidth_gbps * 8  # Convert to Gbps

        results = {
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024**2),
            'avg_latency_us': avg_time * 1e6,  # Convert to microseconds
            'min_latency_us': min_time * 1e6,
            'max_latency_us': max_time * 1e6,
            'std_latency_us': std_time * 1e6,
            'bandwidth_gbps': bandwidth_gbps_network,
            'bandwidth_gb_per_sec': bandwidth_gbps,
            'iterations': timing_iterations
        }

        return results

    return None

def print_benchmark_header():
    if get_rank() == 0:
        print("\n" + "="*90)
        print("MPI Send/Recv Benchmark Results")
        print("="*90)
        print(f"{'Size':<12} {'Size(MB)':<10} {'Avg Lat(μs)':<12} {'Min Lat(μs)':<12} "
              f"{'Max Lat(μs)':<12} {'Std(μs)':<10} {'BW(Gbps)':<12}")
        print("-"*90)

def print_benchmark_result(result):
    if get_rank() == 0 and result:
        size_str = format_size(result['size_bytes'])
        print(f"{size_str:<12} {result['size_mb']:<10.2f} {result['avg_latency_us']:<12.2f} "
              f"{result['min_latency_us']:<12.2f} {result['max_latency_us']:<12.2f} "
              f"{result['std_latency_us']:<10.2f} {result['bandwidth_gbps']:<12.2f}")

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024**2:
        return f"{size_bytes // 1024}KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes // (1024**2)}MB"
    else:
        return f"{size_bytes // (1024**3)}GB"

def main():
    rank = get_rank()
    world_size = get_size()

    if rank == 0:
        print(f"Starting MPI Communication Benchmark")
        print(f"World size: {world_size}")
        print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"Target: 800 Gbps link saturation")

    message_sizes = calculate_message_sizes()

    if rank == 0:
        print(f"Testing {len(message_sizes)} different message sizes")
        print(f"Message size range: {format_size(message_sizes[0])} to {format_size(message_sizes[-1])}")

    print_benchmark_header()

    all_results = []

    for size in message_sizes:
        result = send_recv_benchmark(
            size_bytes=size,
            warmup_iterations=5,
            timing_iterations=10
        )

        if result:
            all_results.append(result)
            print_benchmark_result(result)

    # Print summary
    if rank == 0:
        print("\n" + "="*90)
        print("Benchmark Summary:")

        if all_results:
            # Find peak bandwidth
            peak_bw_result = max(all_results, key=lambda x: x['bandwidth_gbps'])
            print(f"Peak bandwidth: {peak_bw_result['bandwidth_gbps']:.2f} Gbps "
                  f"at {format_size(peak_bw_result['size_bytes'])}")

            # Find minimum latency (for small messages)
            small_msgs = [r for r in all_results if r['size_bytes'] <= 1024]
            if small_msgs:
                min_lat_result = min(small_msgs, key=lambda x: x['min_latency_us'])
                print(f"Minimum latency: {min_lat_result['min_latency_us']:.2f} μs "
                      f"for {format_size(min_lat_result['size_bytes'])}")

            # Check if we achieved target bandwidth
            target_gbps = 800
            achieved_percentage = (peak_bw_result['bandwidth_gbps'] / target_gbps) * 100
            print(f"Link utilization: {achieved_percentage:.1f}% of 800 Gbps target")

        print("="*90)
        print("Benchmark completed successfully!")

# Run the benchmark
if __name__ == "__main__":
    with training_job_context():
        main()

"""


main()

