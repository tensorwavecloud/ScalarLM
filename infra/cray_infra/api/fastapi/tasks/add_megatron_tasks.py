from cray_infra.util.get_config import get_config

from cray_infra.training.restart_megatron_jobs import restart_megatron_jobs
from cray_infra.training.register_megatron_models import register_megatron_models
from cray_infra.generate.clear_acked_requests_from_queue import clear_acked_requests_from_queue

from fastapi_utils.tasks import repeat_every

from contextlib import asynccontextmanager

import traceback
import sys
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def add_megatron_tasks(app):
    config = get_config()

    megatron_refresh_period = config["megatron_refresh_period"]

    @repeat_every(seconds=megatron_refresh_period)
    async def run_megatron_tasks():
        try:
            await register_megatron_models()
            await restart_megatron_jobs()
            await clear_acked_requests_from_queue()
        except Exception as e:
            print_exception()
            raise e

    await run_megatron_tasks()

    yield


def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    messages = traceback.format_exception(exc_type, exc_value, exc_traceback)

    logger.error("".join(messages))
