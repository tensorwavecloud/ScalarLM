from masint.util.make_api_url import make_api_url

import aiohttp
import asyncio

import logging

logger = logging.getLogger(__name__)


def squeue():
    logger.info(f"Getting squeue")

    try:
        asyncio.run(squeue_async())
    except Exception as e:
        logger.error(f"Failed to get squeue output")
        logger.error(e)


async def squeue_async():
    async with aiohttp.ClientSession() as session:
        async with session.get(make_api_url(f"v1/megatron/squeue")) as resp:
            data = await resp.json()

            logger.info(f"Got response for squeue")
            logger.info(data)

            if resp.status != 200:
                logger.error(f"Failed to get squeue")
                logger.error(data)
                raise Exception("Failed to get squeue")

            print(data["squeue_output"])

