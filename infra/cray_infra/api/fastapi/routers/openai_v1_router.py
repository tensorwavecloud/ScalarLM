"""
OpenAI v1 API Router - Standard OpenAI-compatible endpoints.
These endpoints are exposed directly under /v1/ to match OpenAI API spec.
"""

from vllm.entrypoints.openai.protocol import (
    CompletionRequest,
    ChatCompletionRequest,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

import logging

logger = logging.getLogger(__name__)

openai_v1_router = APIRouter()


@openai_v1_router.get("/models")
async def list_models():
    """List available models - proxy to vLLM server."""
    session = get_global_session()
    config = get_config()
    async with session.get(config["vllm_api_url"] + "/v1/models") as resp:
        if resp.status == 200:
            return await resp.json()
        else:
            return JSONResponse(
                content={"error": f"Failed to fetch models: {resp.status}"},
                status_code=resp.status
            )



@openai_v1_router.post("/completions")
async def create_completions(request: CompletionRequest, raw_request: Request):
    """Create completions - proxy to vLLM server."""
    session = get_global_session()
    config = get_config()

    logger.info(f"Received completions request: {request.dict()}")

    allowed_keys = [
        "model",
        "temperature",
        "prompt",
        "max_tokens",
        "stream",
    ]

    params = {
        key: value
        for key, value in request.dict().items()
        if value is not None and key in allowed_keys
    }

    async def generator():
        async with session.post(
            config["vllm_api_url"] + "/v1/completions",
            json=params,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"vLLM completions error ({resp.status}): {error_text}")
                yield f'data: {{"error": "Failed to create completion: {error_text}"}}\n\n'
                return

            async for chunk in resp.content.iter_any():
                yield chunk

    try:
        return StreamingResponse(content=generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completions - proxy to vLLM server."""
    session = get_global_session()
    config = get_config()

    logger.info(f"Received chat completions request: {request.dict()}")

    allowed_keys = [
        "model",
        "temperature",
        "messages",
        "max_tokens",
        "stream",
    ]

    params = {
        key: value
        for key, value in request.dict().items()
        if value is not None and key in allowed_keys
    }

    async def generator():
        async with session.post(
            config["vllm_api_url"] + "/v1/chat/completions",
            json=params,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"vLLM chat completions error ({resp.status}): {error_text}")
                yield f'data: {{"error": "Failed to create chat completion: {error_text}"}}\n\n'
                return

            async for chunk in resp.content.iter_any():
                yield chunk

    try:
        return StreamingResponse(content=generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
