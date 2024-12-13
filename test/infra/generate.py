from cray_infra.one_server.start_cray_server import start_cray_server
from cray_infra.one_server.wait_for_vllm import wait_for_vllm

import masint

import unittest

import logging

logger = logging.getLogger(__name__)


class TestGenerate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["api", "vllm"])

        logger.debug(f"Server started: {self.app}")

    async def test_generate_single(self):
        logger.debug("Testing generate single")

        await wait_for_vllm()

        llm = masint.AsyncSupermassiveIntelligence()

        result = await llm.generate(prompts=["What is 1 + 1?"])

        logger.debug(f"Result: {result}")

    async def test_generate_batch(self):
        logger.debug("Testing generate batch")

        await wait_for_vllm()

        llm = masint.AsyncSupermassiveIntelligence()

        prompts = [
            "What is 1 + 1?",
            "What is 2 + 2?",
            "What is 3 + 3?",
            "What is 4 + 4?",
        ]

        result = await llm.generate(prompts=prompts)

        logger.debug(f"Result: {result}")

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()
