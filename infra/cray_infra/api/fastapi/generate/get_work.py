from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue

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
        first_request, request_id = await inference_work_queue.get()

        if first_request is None:
            return GetWorkResponses(requests=[])

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

            request, request_id = await inference_work_queue.get_nowait()

            if request is None:
                break

            requests.append(
                GetWorkResponse(
                    prompt=request["prompt"],
                    request_id=request_id,
                    model=request["model"],
                    request_type=request["request_type"],
                    max_tokens=request.get("max_tokens", None),
                )
            )

    except Exception as e:
        logger.error(f"Error getting work: {e}")
        logger.error(traceback.format_exc())
        asyncio.sleep(1)

    logger.info(f"Got the following request ids: {[req.request_id for req in requests]}")

    return GetWorkResponses(requests=requests)
