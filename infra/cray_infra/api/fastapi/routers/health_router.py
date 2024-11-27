
from cray_infra.api.fastapi.health.check_health import check_health

from fastapi import APIRouter

import logging

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health")

@health_router.get("")
async def health():
    return await check_health()

@health_router.get("/keepalive")
async def health():
    return {"status": "ok"}
