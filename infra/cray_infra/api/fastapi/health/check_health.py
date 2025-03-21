from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.util.get_config import get_config


async def check_health():
    vllm_health = await get_vllm_health()
    api_health = "up"
    all_health = get_all_health([vllm_health, api_health])
    return {"api": "up", "vllm": vllm_health, "all": all_health}


def get_all_health(healths):
    if all(health == "up" for health in healths):
        return "up"

    if all(health == "down" for health in healths):
        return "down"

    return "mixed"


async def get_vllm_health():
    try:
        session = get_global_session()
        config = get_config()
        async with session.get(config["vllm_api_url"] + "/health") as resp:
            assert resp.status == 200
            return "up"
    except Exception as e:
        return {"status": "down", "reason": str(e)}
