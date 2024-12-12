from cray_infra.util.get_config import get_config

from cray_infra.one_server.start_cray_server import start_cray_server
from cray_infra.one_server.wait_for_vllm import wait_for_vllm

from openai import AsyncOpenAI

import unittest
import asyncio

import logging

logger = logging.getLogger(__name__)


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["api", "vllm"])

        logger.debug(f"Server started: {self.app}")

    async def test_openai_client(self):
        logger.debug("Testing openai client")

        await wait_for_vllm()

        config = get_config()

        client = AsyncOpenAI(
            base_url=config["api_url"] + "/v1/openai",
            api_key="token-abc123",
        )

        completion = await client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10,
        )

        print(completion.choices[0].message)

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()
