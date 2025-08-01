from cray_infra.api.work_queue.inference_work_queue import get_file_backed_inference_work_queue
from cray_infra.util.get_config import get_config

import time
import logging

logger = logging.getLogger(__name__)

async def clear_acked_requests_from_queue():
    inference_work_queue = get_file_backed_inference_work_queue()

    starting_size = len(inference_work_queue)

    await inference_work_queue.clear_acked_data()

    ending_size = len(inference_work_queue)

    logger.info(f"Cleared {starting_size - ending_size} acked requests from the queue.")

    await restart_unacked_requests_from_queue(inference_work_queue)


async def restart_unacked_requests_from_queue(inference_work_queue):
    config = get_config()

    unacked_requests = await inference_work_queue.get_unacked_requests()

    resumed_count = 0

    for request in unacked_requests:

        time_since_submit = time.time() - request["data"]["timestamp"]

        if config["inference_work_queue_ack_timeout"] < time_since_submit:

            await inference_work_queue.resume_unack_task(id=request["pqid"])
            resumed_count += 1

    logger.info(f"Restarted {resumed_count} unacked requests from the queue.")
