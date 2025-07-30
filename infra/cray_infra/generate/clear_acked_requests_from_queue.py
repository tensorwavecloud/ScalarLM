from cray_infra.api.work_queue.inference_work_queue import get_file_backed_inference_work_queue
from cray_infra.util.get_config import get_config
from cray_infra.generate.metrics import get_metrics

import time
import logging

logger = logging.getLogger(__name__)

async def clear_acked_requests_from_queue():
    inference_work_queue = get_file_backed_inference_work_queue()

    starting_size = len(inference_work_queue)

    await inference_work_queue.clear_acked_data()

    ending_size = len(inference_work_queue)

    logger.info(f"Cleared {starting_size - ending_size} acked requests from the queue.")

    config = get_config()

    metrics = get_metrics()
    
    if metrics.epoch_time is not None:
        time_since_epoch = time.time() - metrics.epoch_time

        if config["inference_work_queue_ack_timeout"] < time_since_epoch and metrics.queue_depth > 0:
            unacked_count = await inference_work_queue.unack_count()

            await inference_work_queue.resume_unack_tasks()
            
            logger.info(f"Restarted {unacked_count} unacked requests from the queue.")
