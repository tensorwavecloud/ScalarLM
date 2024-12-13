from cray_infra.one_server.create_api import create_api
from cray_infra.one_server.create_vllm import create_vllm

import asyncio
import logging

logger = logging.getLogger(__name__)


async def start_cray_server(server_list: list):

    running_status = ServerStatus()

    logger.debug(f"Starting servers: {server_list}")

    if ("api" in server_list) or ("all" in server_list):
        logger.debug("Starting API server")
        api_task = asyncio.create_task(
            create_api(port=8000, running_status=running_status)
        )
        running_status.tasks.append(api_task)

    if ("vllm" in server_list) or ("all" in server_list):
        logger.debug("Starting VLLM server")
        vllm_task = asyncio.create_task(
            create_vllm(port=8001, running_status=running_status)
        )
        running_status.tasks.append(vllm_task)

    return running_status


class ServerStatus:
    def __init__(self):
        self.servers = []
        self.tasks = []

    async def shutdown(self):
        for task in self.tasks:
            logger.debug(f"Task {task} is cancelled")
            task.cancel()

        for server in self.servers:
            logger.debug(f"Server {server} is cancelled")
            await server.shutdown()
