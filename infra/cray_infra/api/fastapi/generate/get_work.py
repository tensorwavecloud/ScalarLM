from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.work_queue.get_work_item import get_work_item, get_work_item_no_wait


from cray_infra.api.fastapi.generate.get_adaptors import get_adaptors
from cray_infra.generate.clear_acked_requests_from_queue import worker_ready, worker_not_ready

from cray_infra.api.fastapi.routers.request_types.get_work_request import GetWorkRequest
from cray_infra.api.fastapi.routers.request_types.get_work_response import (
    GetWorkResponses,
    GetWorkResponse,
)

import logging
import traceback
import asyncio

logger = logging.getLogger(__name__)


async def get_work(request: GetWorkRequest):
    batch_size = request.batch_size

    inference_work_queue = await get_inference_work_queue()

    requests = []

    try:
        await worker_ready()
        first_request, request_id = await get_work_item(inference_work_queue)
        await worker_not_ready()

        if first_request is None:
            return GetWorkResponses(requests=[], new_adaptors=await get_adaptors(request))

        requests.append(
            GetWorkResponse(
                prompt=first_request["prompt"],
                request_id=request_id,
                model=first_request["model"],
                request_type=first_request["request_type"],
                max_tokens=first_request.get("max_tokens", None),
            )
        )

        for i in range(batch_size - 1):

            next_request, request_id = await get_work_item_no_wait(inference_work_queue)

            if next_request is None:
                break

            requests.append(
                GetWorkResponse(
                    prompt=next_request["prompt"],
                    request_id=request_id,
                    model=next_request["model"],
                    request_type=next_request["request_type"],
                    max_tokens=next_request.get("max_tokens", None),
                )
            )

    except Exception as e:
        logger.error(f"Error getting work: {e}")
        logger.error(traceback.format_exc())
        await asyncio.sleep(1)

    logger.info(f"Got the following request ids: {[req.request_id for req in requests]}")

    return GetWorkResponses(requests=requests, new_adaptors=await get_adaptors(request))
