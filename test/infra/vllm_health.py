from cray_infra.one_server.start_cray_server import start_cray_server
from cray_infra.one_server.wait_for_vllm import get_health, wait_for_vllm

import unittest

import logging

logger = logging.getLogger(__name__)


class TestVLLMHealth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["vllm"])

        logger.debug(f"Server started: {self.app}")

    async def test_vllm_health(self):
        logger.debug("Testing health endpoint")

        await wait_for_vllm()

        health_status = await get_health()

        self.assertEqual(health_status, 200)

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()
