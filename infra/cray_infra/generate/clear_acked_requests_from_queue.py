from cray_infra.api.work_queue.inference_work_queue import get_file_backed_inference_work_queue

import logging

logger = logging.getLogger(__name__)

async def clear_acked_requests_from_queue():
    inference_work_queue = get_file_backed_inference_work_queue()

    starting_size = len(inference_work_queue)

    await inference_work_queue.clear_acked_data()

    ending_size = len(inference_work_queue)

    logger.info(f"Cleared {starting_size - ending_size} acked requests from the queue.")

