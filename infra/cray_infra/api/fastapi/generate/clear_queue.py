from cray_infra.api.work_queue.inference_work_queue import get_inference_work_queue


async def clear_queue():
    queue = await get_inference_work_queue()

    await queue.clear_queue()
