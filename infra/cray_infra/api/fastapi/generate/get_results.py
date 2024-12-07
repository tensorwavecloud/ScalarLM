from cray_infra.api.fastapi.routers.request_types.get_results_request import GetResultsRequest

from cray_infra.api.fastapi.generate.poll_for_responses import poll_for_responses


async def get_results(request: GetResultsRequest):
    return await poll_for_responses(request.request_ids)
