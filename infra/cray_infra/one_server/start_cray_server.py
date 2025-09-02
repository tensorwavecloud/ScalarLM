from cray_infra.one_server.create_api import create_api
from cray_infra.one_server.create_vllm import create_vllm
from cray_infra.one_server.create_megatron import create_megatron
from cray_infra.one_server.create_generate_worker import create_generate_worker

import asyncio
import logging

logger = logging.getLogger(__name__)


async def start_cray_server(server_list: list):

    server_status = ServerStatus()

    logger.debug(f"Starting servers: {server_list}")

    started_any_server = False

    if ("api" in server_list) or ("all" in server_list):
        logger.debug("Starting API server")
        api_task = asyncio.create_task(
            create_api(port=8000, server_status=server_status)
        )
        server_status.tasks.append(api_task)
        started_any_server = True

    if ("vllm" in server_list) or ("all" in server_list):
        logger.debug("Starting VLLM server")
        vllm_task = asyncio.create_task(
            create_vllm(server_status=server_status, port=8001)
        )
        server_status.tasks.append(vllm_task)
        started_any_server = True

        # Start the generate worker
        logger.debug("Starting Generate Worker")
        worker_task = asyncio.create_task(
            create_generate_worker(server_status=server_status)
        )
        server_status.tasks.append(worker_task)
        logger.info("✓ Generate worker started to process queue requests")

    if "megatron" in server_list:
        logger.debug("Megatron server doesn't need python")
        megatron_task = asyncio.create_task(
            create_megatron(server_status=server_status)
        )
        server_status.tasks.append(megatron_task)
        started_any_server = True

    if not started_any_server:
        logger.error(
            "No valid server type provided. Please specify 'api', 'vllm', 'megatron', or 'all'."
        )

    return server_status


class ServerStatus:
    def __init__(self):
        self.servers = []
        self.tasks = []

        self.app = None
        self.app_event = asyncio.Event()

    async def shutdown(self):
        for task in self.tasks:
            logger.debug(f"Task {task} is cancelled")
            task.cancel()

        for server in self.servers:
            logger.debug(f"Server {server} is cancelled")
            await server.shutdown()

    def set_app(self, app):
        self.app = app
        logger.debug(f"VLLM app set: {app}")
        self.app_event.set()

    async def get_app(self):
        if self.app is None:
            logger.debug("Waiting for VLLM app to be set")
            await self.app_event.wait()

        logger.debug(f"Returning VLLM app: {self.app}")
        return self.app
