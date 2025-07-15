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
from gpu_aware_mpi import get_size, get_rank, allgather, barrier
import torch

from cray_infra.training.training_job_context import training_job_context

message_sizes = [2 ** i for i in range(10)]

def allgather_test(size):
    world_size = get_size()

    input_tensor = torch.ones(size // 4, dtype=torch.float32)
    output_tensor = torch.ones(world_size * size // 4, dtype=torch.float32)

    allgather(input_tensor, output_tensor)

    barrier()

    if get_rank() == 0:
        print(f"Allgather completed for size {size} with data: {input_tensor}")

with training_job_context():
    for size in message_sizes:
        allgather_test(size)

    if get_rank() == 0:
        print("All allgather tests completed successfully.")

"""


main()

