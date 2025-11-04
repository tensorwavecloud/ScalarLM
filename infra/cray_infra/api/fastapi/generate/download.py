from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock
from cray_infra.api.work_queue.group_request_id_to_path import group_request_id_to_path
from cray_infra.api.work_queue.group_request_id_to_status_path import group_request_id_to_status_path
from cray_infra.api.work_queue.group_request_id_to_response_path import group_request_id_to_response_path

from fastapi.responses import FileResponse

import json


async def download(download_request):
    request_id = download_request.request_id

    config = get_config()

    timeout = config["response_timeout"]

    start_time = time.time()

    while time.time() - start_time < timeout:
        file_path = group_request_id_to_path(request_id)
        try:
            async with acquire_file_lock(file_path):
                status_path = group_request_id_to_status_path(file_path)
                with open(status_path, "r") as status_file:
                    status = json.load(status_file)
                    if status["status"] == "completed":
                        response_path = group_request_id_to_response_path(file_path)
                        return FileResponse(
                            response_path,
                            media_type="application/json",
                            filename=f"{request_id}_response.json",
                        )
        except FileNotFoundError:
            await asyncio.sleep(0.1)
