from cray_infra.util.get_config import get_config

from cray_infra.training.register_megatron_workers import register_megatron_workers

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from fastapi_utils.tasks import repeat_every

import traceback
import sys
import logging

logger = logging.getLogger(__name__)

async def create_megatron(server_status):
    server_config = uvicorn.Config(
        "cray_infra.one_server.create_megatron:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    server_status.servers.append(server)

    await server.serve()

@asynccontextmanager
async def add_megatron_tasks(app):
    config = get_config()

    megatron_refresh_period = config["megatron_refresh_period"]

    @repeat_every(seconds=megatron_refresh_period)
    async def run_megatron_tasks():
        try:
            await register_megatron_workers()
        except Exception as e:
            print_exception()
            raise e

    await run_megatron_tasks()

    yield


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    messages = traceback.format_exception(exc_type, exc_value, exc_traceback)

    logger.error("".join(messages))

app = FastAPI(lifespan=add_megatron_tasks)
