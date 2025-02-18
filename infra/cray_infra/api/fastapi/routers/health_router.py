from cray_infra.api.fastapi.health.check_health import check_health

from fastapi import APIRouter

from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

health_router = APIRouter(prefix="/health")


@health_router.get("")
async def health():
    return await check_health()


@health_router.get("/keepalive")
async def health():
    return {"status": "ok"}


@health_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in health_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
