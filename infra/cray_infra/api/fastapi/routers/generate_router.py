from cray_infra.api.fastapi.generate.get_work import get_work
from cray_infra.api.fastapi.generate.generate import generate
from cray_infra.api.fastapi.generate.finish_work import finish_work
from cray_infra.api.fastapi.generate.get_results import get_results
from cray_infra.api.fastapi.generate.embed import embed
from cray_infra.api.fastapi.generate.metrics import metrics

from cray_infra.api.fastapi.routers.request_types.embed_request import EmbedRequest

from cray_infra.api.fastapi.routers.request_types.generate_request import (
    GenerateRequest,
)
from cray_infra.api.fastapi.routers.request_types.get_work_request import GetWorkRequest
from cray_infra.api.fastapi.routers.request_types.finish_work_request import (
    FinishWorkRequests,
)
from cray_infra.api.fastapi.routers.request_types.get_results_request import (
    GetResultsRequest,
)

from fastapi import APIRouter
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

generate_router = APIRouter(prefix="/generate")


@generate_router.post("")
async def generate_endpoint(request: GenerateRequest):
    return await generate(request)


@generate_router.post("/get_results")
async def get_results_endpoint(request: GetResultsRequest):
    return await get_results(request)


@generate_router.post("/get_work")
async def get_work_endpoint(request: GetWorkRequest):
    return await get_work(request)


@generate_router.post("/finish_work")
async def finish_work_endpoint(requests: FinishWorkRequests):
    return await finish_work(requests)


@generate_router.post("/embed")
async def embed_endpoint(request: EmbedRequest):
    return await embed(request)


@generate_router.get("/metrics")
async def metrics_endpoint():
    return await metrics()


@generate_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in generate_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
