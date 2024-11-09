from cray_infra.training.training_job_status import TrainingJobStatus
from cray_infra.training.launch_training_job import start_slurm_job

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

    # Filter out the jobs that are already running
    jobs = filter_running_jobs(all_jobs, slurm_job_names)

    # Restart all the jobs
    async for job in jobs:
        await restart_job(job)


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
    squeue_output = subprocess.check_output(["squeue", "-o", "%i", "-h"])

    # Here is an example of the output of squeue
    # JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    # 1234      gpu  jobname  username  R       0:10      1 node1
    # 5678      gpu  jobname  username  R       0:10      1 node2
    # 91011      gpu  jobname  username  R       0:10      1 node3
    # 121314      gpu  jobname  username  R       0:10      1 node4

    # We want to get the job name from the third column
    # We also want to remove the header
    jobs = squeue_output.decode().split("\n")[1:]

    job_names = [job.split()[2] for job in jobs if job]

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
