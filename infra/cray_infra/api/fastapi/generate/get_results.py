from cray_infra.api.fastapi.routers.request_types.get_results_request import GetResultsRequest
from cray_infra.api.fastapi.routers.request_types.get_results_response import GetResultsResponse
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id

from cray_infra.api.fastapi.generate.poll_for_responses import poll_for_responses


async def get_results(request: GetResultsRequest):

    unique_group_request_ids = set()

    for request_id in request.request_ids:
        group_request_id = get_group_request_id(request_id)
        unique_group_request_ids.add(group_request_id)

    all_responses = []

    for group_request_id in unique_group_request_ids:
        responses = await poll_for_responses(group_request_id)

        all_responses.extend(responses.results)

    return GetResultsResponse(results=all_responses)
