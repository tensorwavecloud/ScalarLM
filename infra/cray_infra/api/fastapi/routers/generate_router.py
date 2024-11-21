from cray_infra.api.fastapi.generate.generate import generate

from cray_infra.api.fastapi.routers.request_types.generate_request import GenerateRequest

from fastapi import APIRouter

import logging

logger = logging.getLogger(__name__)

generate_router = APIRouter(prefix="/generate")

@generate_router.post("")
async def generate_endpoint(request : GenerateRequest):
    return await generate(request)

