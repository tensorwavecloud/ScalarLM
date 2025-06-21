from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequests,
)
from cray_infra.generate.metrics import get_metrics

import logging

logger = logging.getLogger(__name__)


async def finish_work(requests: FinishWorkRequests):
    inference_work_queue = await get_inference_work_queue()

    for request in requests.requests:
        logger.debug(f"Finishing work for request {request.request_id}")

        result = await inference_work_queue.get_id(id=request.request_id)

        if request.response is not None:
            result["response"] = request.response

        if request.error is not None:
            result["error"] = request.error

        await inference_work_queue.update_and_ack(id=request.request_id, item=result)

        metrics = get_metrics()

        metrics.record_completed_request(
            token_count=request.token_count, flop_count=request.flop_count
        )
