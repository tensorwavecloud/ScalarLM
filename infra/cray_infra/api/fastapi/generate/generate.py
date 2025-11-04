from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
    Result,
)

from cray_infra.api.fastapi.generate.poll_for_responses import poll_for_responses
from cray_infra.api.fastapi.generate.upload import get_request_path
from cray_infra.api.work_queue.push_into_queue import push_into_queue

from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.generate.metrics import get_metrics

from cray_infra.util.get_config import get_config

from fastapi import HTTPException

import json
import traceback
import hashlib

import logging

logger = logging.getLogger(__name__)


async def generate(request: GenerateRequest):

    prompts = request.prompts
    model = request.model
    max_tokens = request.max_tokens

    logger.info(
        f"Received generate request: prompts={truncate_list(prompts)}, "
        f"model={model}, max_tokens={max_tokens}"
    )

    config = get_config()

    if model is None:
        model = config["model"]
        logger.info(f"Using default model: {model}")

    if model == "latest":
        model = get_latest_model()

    model_manager = get_vllm_model_manager()

    model = model_manager.find_model(model)

    if model is None:
        logger.error(f"Model {model} not found")
        raise HTTPException(status_code=404, detail=f"Model {model} not found")

    requests = []

    try:
        request_count = 0

        for prompt in prompts:
            request = (
                {
                    "prompt": prompt,
                    "model": model,
                    "max_tokens": max_tokens,
                    "request_type": "generate",
                }
            )

            requests.append(request)

            metrics = get_metrics()

            metrics.record_new_request()
            request_count += 1

        contents_hash = get_contents_hash(requests)
        request_id = contents_hash.hexdigest()

        request_path = get_request_path(request_id)

        logger.info(f"Generated request path: {request_path}")

        with open(request_path, "w") as f:
            json.dump(requests, f)

        await push_into_queue(request_count, request_path)

    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating responses: {e}")

    logger.info(f"Generated request_id: {request_id}")

    try:
        responses = await poll_for_responses(request_id)
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        logger.error(traceback.format_exc())
        responses = GenerateResponse(
            results=[
                Result(request_id=f"{request_id}_{index}", response=None)
                for index, request in enumerate(requests)
            ]
        )

    logger.info(f"Generated responses: {truncate_list(responses)}")
    return responses


def truncate_list(list_of_strings):
    return [truncate_string(str(s)) for s in list_of_strings]


def truncate_string(s):
    if len(s) > 100:
        return s[:100] + "..."
    else:
        return s

def get_contents_hash(requests) -> hashlib._hashlib.HASH:
    sha256 = hashlib.sha256()
    sha256.update(json.dumps(requests).encode("utf-8"))
    return sha256
