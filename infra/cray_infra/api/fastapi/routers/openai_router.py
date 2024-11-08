from vllm.entrypoints.openai.protocol import TokenizeRequest

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from fastapi import APIRouter, Request

import logging

logger = logging.getLogger(__name__)

openai_router = APIRouter(prefix="/openai")


@openai_router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    session = get_global_session()

    async with session.post("http://localhost:8001/tokenize", json=request.dict()) as resp:
        assert resp.status == 200
        return await resp

