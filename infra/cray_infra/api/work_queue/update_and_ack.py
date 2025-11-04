from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.group_request_id_to_response_path import (
    group_request_id_to_response_path,
)
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id
from cray_infra.api.work_queue.get_in_memory_results import (
    get_in_memory_results,
    clear_in_memory_results,
)

import time
import json
import logging

logger = logging.getLogger(__name__)

async def update_and_ack(inference_work_queue, request_id, item):
    logger.info(f"Acknowledging request {request_id}")

    group_request_id = get_group_request_id(request_id)
    in_memory_results = await get_in_memory_results(group_request_id)

    if in_memory_results["results"][request_id]["is_acked"]:
        logger.warn(f"Request {id} is already acknowledged")
    else:
        in_memory_results["current_index"] += 1

    in_memory_results["results"][request_id] = item
    in_memory_results["results"][request_id]["is_acked"] = True

    if in_memory_results["current_index"] >= in_memory_results["total_requests"]:
        await finish_work_queue_item(request_id, inference_work_queue, in_memory_results)


async def finish_work_queue_item(request_id, inference_work_queue, in_memory_results):
    group_request_id = get_group_request_id(request_id)
    response_path = group_request_id_to_response_path(group_request_id)

    async with acquire_file_lock(response_path):

        # Save results to disk
        with open(response_path, "w") as response_file:
            json.dump(in_memory_results, response_file)

        # Update the status file
        status_path = group_request_id_to_status_path(group_request_id)

        with open(status_path, "r") as status_file:
            current_status = json.load(status_file)

        current_status["status"] = "completed"
        current_status["completed_at"] = time.time()
        current_status["current_index"] = in_memory_results["current_index"]

        with open(status_path, "w") as status_file:
            json.dump(current_status, status_file)

    logger.info(f"Finished processing group request {group_request_id}")
    logger.info(f"Acknowledging work queue item {current_status['work_queue_id']}")

    await inference_work_queue.ack(current_status["work_queue_id"])

    clear_in_memory_results(group_request_id)
