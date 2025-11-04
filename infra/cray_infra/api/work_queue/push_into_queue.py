from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue
from cray_infra.api.work_queue.get_work_item import strip_request_id
from cray_infra.api.work_queue.group_request_id_to_status_path import group_request_id_to_status_path

import json

async def push_into_queue(request_count, item_path):
    inference_work_queue = await get_inference_work_queue()

    id = await inference_work_queue.put({"path": item_path, "request_count": request_count})

    group_request_id = strip_request_id(item_path)

    status_file_path = group_request_id_to_status_path(group_request_id)

    with open(status_file_path, "w") as status_file:
        status = {
            "status": "in_progress",
            "current_index": 0,
            "total_requests": request_count,
            "work_queue_id": id,
        }

        json.dump(status, status_file)
