from cray_infra.util.get_config import get_config

import asyncio
import aiohttp

import logging

logger = logging.getLogger(__name__)


async def wait_for_vllm():
    for _ in range(30):
        health_status = await get_vllm_health()
        if health_status == 200:
            return
        await asyncio.sleep(1)


async def get_vllm_health():
    config = get_config()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config["vllm_api_url"] + "/health") as response:
                return response.status
    except Exception as e:
        logger.error(f"Error getting health: {e}")
        return 500
