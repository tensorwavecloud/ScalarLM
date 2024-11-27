from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.launch_training_job import start_slurm_job

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.util.get_config import get_config


import os
import json
import yaml
import subprocess

import logging

logger = logging.getLogger(__name__)


async def restart_megatron_jobs():
    logger.info("Restarting Megatron jobs")

    # Get all the jobs that are running
    all_jobs = get_running_jobs()

    # Get slurm jobs that are running
    slurm_job_names = await get_slurm_jobs()

    logger.info(f"Slurm jobs running: {slurm_job_names}")

    # Filter out the jobs that are already running
    jobs = filter_running_jobs(all_jobs, slurm_job_names)

    # Restart all the jobs
    async for job in jobs:
        await restart_job(job)

    # Get slurm jobs that are running again
    slurm_job_names = await get_slurm_jobs()

    # If any are still running, keep the server alive
    if slurm_job_names:
        logger.info("Jobs are still running, keeping the server alive")
        await keep_alive()


async def get_running_jobs():
    config = get_config()

    # training jobs are in the training job directory
    # they are subdirectories which should have a file called status.json

    # get all the directories in the training job directory
    # that have a status.json file

    for root, dirs, files in os.walk(config["training_job_directory"]):
        if "status.json" in files:
            with open(os.path.join(root, "status.json")) as f:
                status = json.load(f)
                if (
                    status["status"] == TrainingJobStatus.TRAINING
                    or status["status"] == TrainingJobStatus.QUEUED
                ):
                    yield root


async def get_slurm_jobs():
    squeue_output = subprocess.check_output(
        ["squeue", '--format="%.18i %.9P %.128j %.8u %.8T %.10M %.9l %.6D %R"']
    )

    # Here is an example of the output of squeue
    # JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    # 1234      gpu  jobname  username  R       0:10      1 node1
    # 5678      gpu  jobname  username  R       0:10      1 node2
    # 91011      gpu  jobname  username  R       0:10      1 node3
    # 121314      gpu  jobname  username  R       0:10      1 node4

    # We want to get the job name from the third column
    # We also want to remove the header
    jobs = squeue_output.decode().split("\n")[1:]

    job_names = []

    for job in jobs:
        if job:
            job_fields = job.strip().split()
            if len(job_fields) > 3:
                job_names.append(job_fields[3])

    return job_names


async def filter_running_jobs(all_jobs, slurm_job_names):
    async for job in all_jobs:
        if os.path.basename(job) not in slurm_job_names:
            yield job


async def restart_job(job):
    logger.info(f"Restarting job: {job}")

    # Get the job config
    with open(os.path.join(job, "config.yaml")) as f:
        config = yaml.safe_load(f)

    start_slurm_job(config)


async def keep_alive():
    config = get_config()
    session = get_global_session()
    try:
        async with session.get(config["api_url"] + "/v1/health/keepalive") as resp:
            assert resp.status == 200
    except Exception as e:
        logger.error(f"Error keeping the server alive: {e}")
