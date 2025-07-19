from cray_infra.api.fastapi.routers.request_types.get_adaptors_response import (
    GetAdaptorsResponse,
)

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

async def get_adaptors(request):

    already_loaded_adaptor_count = request.loaded_adaptor_count

    vllm_model_manager = get_vllm_model_manager()

    registered_models = vllm_model_manager.get_registered_models()

    new_adaptors = registered_models[already_loaded_adaptor_count:]

    return GetAdaptorsResponse(new_adaptors=new_adaptors)
