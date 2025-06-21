from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue

from cray_infra.api.fastapi.routers.request_types.generate_response import Result
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
)

from cray_infra.util.get_config import get_config

import time
import asyncio
import copy

import logging

logger = logging.getLogger(__name__)


async def poll_for_responses(request_ids):

    config = get_config()

    timeout = config["response_timeout"]

    inference_work_queue = await get_inference_work_queue()

    start_time = time.time()

    responses = GenerateResponse(results=[])

    responses_so_far = set()

    while len(responses.results) < len(request_ids):
        if time.time() - start_time > timeout:
            break

        for request_id in request_ids:
            if request_id in responses_so_far:
                continue

            response = await inference_work_queue.get_if_finished(request_id)

            if response is None:
                continue

            logger.info("Received response for request_id: "
                f"{request_id} - {truncate_fields(response)}")

            responses_so_far.add(request_id)
            responses.results.append(
                Result(
                    request_id=request_id,
                    response=response.get("response", None),
                    error=response.get("error", None),
                )
            )

        await asyncio.sleep(0.1)

    # set the response to None for any requests that did not finish
    for request_id in request_ids:
        if request_id not in responses_so_far:
            logger.info(f"Request {request_id} did not finish in time")
            responses.results.append(Result(request_id=request_id, response=None))

    # Sort the responses by request_id, with None responses at the end
    responses.results.sort(key=lambda x: (x.request_id is None, x.request_id))

    return responses

def truncate_fields(response):
    # Limit the length of the prompt and error fields to 100 characters
    response = copy.deepcopy(response)
    response["prompt"] = response.get("prompt", "")[:100]
    response["error"] = response.get("error", "")[:100]
    return response
