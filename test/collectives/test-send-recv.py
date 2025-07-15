import scalarlm

import logging

scalarlm.api_url = "http://localhost:8000"


def main():
    llm = scalarlm.SupermassiveIntelligence()

    gpu_count = max(2, llm.get_gpu_count())

    status = llm.submit_slurm_job(
        code=get_code(), train_args={"gpus": gpu_count, "max_gpus": gpu_count}
    )

    print(status)


def get_code():
    return """
from gpu_aware_mpi import get_size, get_rank, send, recv, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

message_sizes = [2 ** i for i in range(10)]

def send_recv_test(size):
    my_tensor = torch.ones(size // 4, dtype=torch.float32)

    # cross the network bisection
    rank = get_rank()
    bisection_rank = get_size() // 2

    if bisection_rank == 0:
        print("Bisection rank is 0, skipping test to avoid deadlock.")
        return

    neighbor = (rank + bisection_rank) % get_size()

    if rank < bisection_rank:
        send(my_tensor, neighbor)
        print(f"Rank {rank} sent tensor of size {size} to rank {neighbor}")
    else:
        received_tensor = recv(my_tensor, neighbor)
        print(f"Rank {rank} received tensor of size {size} from rank {neighbor}")

    barrier()

with training_job_context():
    for size in message_sizes:
        send_recv_test(size)

    if get_rank() == 0:
        print("All send/recv tests completed successfully.")


"""


main()
