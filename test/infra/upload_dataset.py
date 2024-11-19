from cray_infra.one_server.start_cray_server import start_cray_server

import masint

import unittest

import logging

logger = logging.getLogger(__name__)


class TestUploadDataset(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):

        logger.info("Starting server")

        self.app = await start_cray_server(server_list=["api"])

        logger.debug(f"Server started: {self.app}")

    async def test_upload_dataset(self):
        logger.debug("Testing upload ability of train endpoint")

        llm = masint.AsyncSupermassiveIntelligence()

        dataset = get_dataset()

        status = await llm.train(dataset, train_args={"max_steps": 1})

    async def asyncTearDown(self):
        logger.debug("Shutting down server")
        await self.app.shutdown()


def get_dataset():
    dataset = []

    count = 10000

    for i in range(count):
        dataset.append(
            {"input": f"What is {i} + {i}", "output": "The answer is " + str(i + i)}
        )

    return dataset
