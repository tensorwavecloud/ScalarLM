import scalarlm

import logging

logging.basicConfig(level=logging.DEBUG)

scalarlm.api_url = "http://localhost:8000"


def main():
    llm = scalarlm.SupermassiveIntelligence()

    gpu_count = llm.get_gpu_count()

    status = llm.submit_slurm_job(code=get_code(), train_args={"gpus": gpu_count})

    print(status)


def get_code():
    return """
from gpu_aware_mpi import get_size, get_rank
from cray_infra.training.training_job_context import training_job_context

with training_job_context():
    print(f"Hello from rank {get_rank()} of {get_size()}")

"""


main()
