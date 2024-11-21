from vllm.entrypoints.openai.protocol import (
    TokenizeRequest,
    ChatCompletionRequest,
    CompletionRequest,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse

import logging

logger = logging.getLogger(__name__)

openai_router = APIRouter(prefix="/openai")


@openai_router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    session = get_global_session()
    config = get_config()

    async with session.post(
        config["vllm_api_url"] + "/tokenize", json=request.dict()
    ) as resp:
        assert resp.status == 200
        return await resp.json()


@openai_router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    session = get_global_session()

    logger.info(f"Received request: {request.dict()}")
    logger.info(f"Received raw request: {raw_request.json()}")

    allowed_keys = [
        "model",
        "temperature",
        "messages",
        "max_tokens",
    ]

    params = {
        key: value
        for key, value in request.dict().items()
        if value is not None and key in allowed_keys
    }

    params["temperature"] = 0.0

    logger.info(f"Sending request: {params}")

    config = get_config()

    async def generator():
        async with session.post(
            config["vllm_api_url"] + "/v1/chat/completions",
            json=params,
        ) as resp:
            assert resp.status == 200

            async for chunk in resp.content.iter_any():
                yield chunk

    try:
        return StreamingResponse(content=generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@openai_router.post("/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    session = get_global_session()

    logger.info(f"Received request: {request.dict()}")
    logger.info(f"Received raw request: {raw_request.json()}")

    allowed_keys = [
        "model",
        "temperature",
        "prompt",
        "max_tokens",
    ]

    params = {
        key: value
        for key, value in request.dict().items()
        if value is not None and key in allowed_keys
    }

    params["temperature"] = 0.0

    logger.info(f"Sending request: {params}")

    config = get_config()

    async def generator():
        async with session.post(
            config["vllm_api_url"] + "/v1/completions",
            json=params,
        ) as resp:
            assert resp.status == 200

            async for chunk in resp.content.iter_any():
                yield chunk

    try:
        return StreamingResponse(content=generator(), media_type="text/event-stream")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
