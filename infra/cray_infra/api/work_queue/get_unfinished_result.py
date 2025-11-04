from cray_infra.api.work_queue.get_in_memory_results import get_in_memory_results
from cray_infra.api.work_queue.get_group_request_id import get_group_request_id

async def get_unfinished_result(request_id):
    group_request_id = get_group_request_id(request_id)
    in_memory_results = await get_in_memory_results(group_request_id)

    if not request_id in in_memory_results["results"]:
        in_memory_results["results"][request_id] = { "is_acked": False }

    return in_memory_results["results"][request_id]
