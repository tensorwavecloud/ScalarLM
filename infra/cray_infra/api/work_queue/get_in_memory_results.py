import asyncio

lock = asyncio.Lock()
in_memory_results = {}

async def get_in_memory_results(group_request_id):
    global in_memory_results
    global lock

    async with lock:
        if group_request_id not in in_memory_results:
            in_memory_results[group_request_id] = {
                "results": {},
                "current_index": 0,
                "total_requests": 0,
                "work_queue_id": None,
            }

        return in_memory_results[group_request_id]

async def clear_in_memory_results(group_request_id):
    global in_memory_results
    global lock

    async with lock:
        if group_request_id in in_memory_results:
            del in_memory_results[group_request_id]

