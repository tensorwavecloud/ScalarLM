from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.generate_response import (
    GenerateResponse,
)

from cray_infra.util.get_config import get_config

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from fastapi import HTTPException

import json

import logging

logger = logging.getLogger(__name__)


async def generate(request: GenerateRequest):
    logger.info(f"Received generate request: {request}")

    prompts = request.prompts
    model = request.model

    config = get_config()

    if model is None:
        model = config["model"]
        logger.info(f"Using default model: {model}")

    # Group the prompts into batches
    prompt_batches = []

    for i in range(0, len(prompts), config["generate_batch_size"]):
        prompt_batches.append(prompts[i : i + config["generate_batch_size"]])

    # Generate the responses
    try:
        responses = []
        for prompt_batch in prompt_batches:
            response_batch = await async_submit_generate_request(prompt_batch, model)

            responses.extend(response_batch)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Generated responses: {responses}")
    return GenerateResponse(responses=responses)


async def async_submit_generate_request(prompts, model):
    config = get_config()
    url = f"{config['api_url']}/v1/openai/completions"

    logger.info(f"Submitting generate request to {url}")

    responses = []
    session = get_global_session()
    for prompt in prompts:
        json_request = {"prompt": prompt, "model": model}

        logger.info(f"Sending request: {json_request}")
        async with session.post(url, json=json_request) as resp:
            response_text = ""
            async for chunk in resp.content.iter_any():
                logger.info(f"Received chunk: {chunk}")
                response_text += chunk.decode("utf-8")

            logger.info(f"Received response: {response_text}")

            response = json.loads(response_text)

            responses.append(response["choices"][0]["text"])

    return responses
