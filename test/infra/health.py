import logging

import aiohttp
import unittest
import pytest

from cray_infra.one_server.start_cray_server import start_cray_server
from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


class TestHealth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["api"])

        logger.debug(f"Server started: {self.app}")

    async def test_webserver_get(self):
        logger.debug("Testing health endpoint")
        health_status = await get_health()

        self.assertEqual(health_status["api"], "up")

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()


async def get_health():
    config = get_config()

    async with aiohttp.ClientSession() as session:
        async with session.get(config["api_url"] + "/v1/health") as response:
            return await response.json()
