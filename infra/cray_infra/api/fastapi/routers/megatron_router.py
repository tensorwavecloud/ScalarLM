from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse,
)

from cray_infra.training.launch_training_job import launch_training_job
from cray_infra.training.upload_training_data import upload_training_data
from cray_infra.training.training_logs_generator import training_logs_generator
from cray_infra.training.get_training_job_info import get_training_job_info
from cray_infra.training.list_models import list_models
from cray_infra.training.squeue import squeue

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse

import traceback

import logging

logger = logging.getLogger(__name__)

megatron_router = APIRouter(prefix="/megatron")


@megatron_router.post("/train")
async def train(request: Request):
    logger.info(f"Training request received: {request}")
    training_data_path, params = await upload_training_data(request)

    try:
        job_config = params

        logger.info(f"Training args: {job_config}")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid request body",
        )

    job_status = await launch_training_job(job_config)

    return TrainResponse(job_status=job_status, job_config=job_config, deployed=False)


@megatron_router.get("/train/{job_hash}")
async def job_info(job_hash: str):
    return await get_training_job_info(job_hash)


@megatron_router.get("/train/logs/{model_name}")
async def get_training_logs(model_name: str, starting_line_number: int = 0):
    try:
        return StreamingResponse(
            content=training_logs_generator(model_name, starting_line_number),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(e)
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@megatron_router.get("/list_models")
async def models():
    return await list_models()


@megatron_router.get("/squeue")
async def get_squeue():
    return await squeue()


@megatron_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in megatron_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
