from cray_infra.api.fastapi.routers.request_types.get_adaptors_response import (
    GetAdaptorsResponse,
)

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.util.get_config import get_config

import time

import asyncio


async def get_adaptors(request):

    already_loaded_adaptor_count = request.loaded_adaptor_count

    vllm_model_manager = get_vllm_model_manager()

    config = get_config()

    get_adaptors_timeout = config.get("get_adaptors_timeout", 30)
    get_adaptors_poll_interval = config.get("get_adaptors_poll_interval", 0.1)
    start_time = time.time()

    while time.time() - start_time < get_adaptors_timeout:
        registered_models = vllm_model_manager.get_registered_models()

        new_adaptors = registered_models[already_loaded_adaptor_count:]

        if len(new_adaptors) > 0:
            # If there are new adaptors, return them immediately
            break

        await asyncio.sleep(get_adaptors_poll_interval)

    return GetAdaptorsResponse(new_adaptors=new_adaptors)
