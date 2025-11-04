from cray_infra.api.work_queue.inference_work_queue import get_file_backed_inference_work_queue
from cray_infra.util.get_config import get_config

import time
import logging

logger = logging.getLogger(__name__)

ready_worker_idle_start_time = None


async def clear_acked_requests_from_queue():
    inference_work_queue = get_file_backed_inference_work_queue()

    starting_size = len(inference_work_queue)

    await inference_work_queue.clear_acked_data()

    ending_size = len(inference_work_queue)

    logger.info(f"Cleared {starting_size - ending_size} acked requests from the queue.")

    await restart_unacked_requests_from_queue(inference_work_queue)


async def restart_unacked_requests_from_queue(inference_work_queue):
    global ready_worker_idle_start_time

    config = get_config()

    unacked_requests = await inference_work_queue.get_unacked_requests()

    resumed_count = 0

    for request in unacked_requests:

        time_since_submit = time.time() - request["data"]["timestamp"]
        ready_worker_idle_time = (
            0
            if ready_worker_idle_start_time is None
            else time.time() - ready_worker_idle_start_time
        )

        if (config["inference_work_queue_ack_timeout"] < time_since_submit) and (
            ready_worker_idle_time > config["inference_work_queue_idle_time"]
        ):
            await inference_work_queue.resume_unack_task(id=request["id"])
            resumed_count += request["data"]["request_count"]

    logger.info(f"Restarted {resumed_count} unacked requests from the queue.")


async def worker_ready():
    global ready_worker_idle_start_time
    ready_worker_idle_start_time = time.time()


async def worker_not_ready():
    global ready_worker_idle_start_time
    ready_worker_idle_start_time = None
