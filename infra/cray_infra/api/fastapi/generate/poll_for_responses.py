from cray_infra.api.work_queue.get_in_memory_results import get_in_memory_results
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id
from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)

from cray_infra.api.fastapi.routers.request_types.generate_response import Result
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
)

from cray_infra.util.get_config import get_config

import time
import asyncio
import copy
import json

import logging

logger = logging.getLogger(__name__)


async def poll_for_responses(group_request_id):

    request_ids = get_request_ids(group_request_id)

    config = get_config()

    timeout = config["response_timeout"]

    start_time = time.time()

    responses = GenerateResponse(results=[])

    responses_so_far = set()

    while len(responses.results) < len(request_ids):
        if time.time() - start_time > timeout:
            break

        for request_id in request_ids:
            if request_id in responses_so_far:
                continue

            group_request_id = get_group_request_id(request_id)
            in_memory_results = await get_in_memory_results(group_request_id)

            if request_id not in in_memory_results["results"]:
                continue

            response = in_memory_results["results"][request_id]

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
    response["prompt"] = str(response.get("prompt", ""))[:100]
    response["error"] = str(response.get("error", ""))[:100]
    return response

def get_request_ids(group_request_id):
    status_file_path = group_request_id_to_status_path(group_request_id)

    with open(status_file_path, "r") as status_file:
        status = json.load(status_file)

    total_requests = status["total_requests"]

    request_ids = [f"{group_request_id}_{i}" for i in range(total_requests)]

    return request_ids
