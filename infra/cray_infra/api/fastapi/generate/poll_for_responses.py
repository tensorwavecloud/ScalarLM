from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue

from cray_infra.api.fastapi.routers.request_types.generate_response import Result
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
)

from cray_infra.util.get_config import get_config

import time
import asyncio

import logging

logger = logging.getLogger(__name__)


async def poll_for_responses(request_ids):

    config = get_config()

    timeout = config["response_timeout"]

    inference_work_queue = get_inference_work_queue()

    start_time = time.time()

    responses = GenerateResponse(results=[])

    responses_so_far = set()

    while len(responses.results) < len(request_ids):
        if time.time() - start_time > timeout:
            break

        for request_id in request_ids:
            if request_id in responses_so_far:
                continue

            response = inference_work_queue.get_if_finished(request_id)

            if response is None:
                continue

            logger.info(f"Received response for request_id: {request_id} - {response}")

            responses_so_far.add(request_id)
            responses.results.append(
                Result(request_id=request_id, response=response["response"])
            )

        await asyncio.sleep(0.1)

    # set the response to None for any requests that did not finish
    for request_id in request_ids:
        if request_id not in responses_so_far:
            responses.results.append(Result(request_id=request_id, response=None))

    return responses
