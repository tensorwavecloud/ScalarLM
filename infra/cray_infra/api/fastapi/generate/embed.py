from cray_infra.api.fastapi.routers.request_types.embed_request import (
    EmbedRequest,
)
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
    Result,
)

from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.fastapi.generate.poll_for_responses import poll_for_responses

from cray_infra.util.get_config import get_config

from fastapi import HTTPException

import json
import traceback

import logging

logger = logging.getLogger(__name__)


async def embed(request: EmbedRequest):

    prompts = request.prompts
    model = request.model

    logger.info(
        f"Received embed request: prompts={truncate_list(prompts)}, "
        f"model={model}"
    )

    config = get_config()

    if model is None:
        model = config["model"]
        logger.info(f"Using default model: {model}")

    inference_work_queue = await get_inference_work_queue()

    request_ids = []

    try:
        for prompt in prompts:
            request_id = await inference_work_queue.put(
                {"prompt": prompt, "model": model, "request_type": "embed"}
            )

            request_ids.append(request_id)

    except Exception as e:
        logger.error(f"Error generating embedding responses: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating embedding responses: {e}")

    logger.info(f"Generated request_ids: {request_ids}")

    try:
        responses = await poll_for_responses(request_ids)
    except Exception as e:
        logger.error(f"Error generating embedding responses: {e}")
        logger.error(traceback.format_exc())
        responses = GenerateResponse(
            results=[
                Result(request_id=request_id, response=None)
                for request_id in request_ids
            ]
        )

    logger.info(f"Generated embedding responses: {responses}")
    return responses


def truncate_list(list_of_strings):
    return [truncate_string(s) for s in list_of_strings]


def truncate_string(s):
    if len(s) > 100:
        return s[:100] + "..."
    else:
        return s

