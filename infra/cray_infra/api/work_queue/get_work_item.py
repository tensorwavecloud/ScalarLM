from cray_infra.api.work_queue.acquire_file_lock import acquire_file_lock

from cray_infra.api.work_queue.group_request_id_to_status_path import (
    group_request_id_to_status_path,
)

import asyncio
import json
import os

import logging

logger = logging.getLogger(__name__)

lock = asyncio.Lock()
in_memory_work_queue = []


async def get_work_item(work_queue):
    global in_memory_work_queue
    global lock

    async with lock:
        if not in_memory_work_queue:
            await fill_work_queue(work_queue)

        if not in_memory_work_queue:
            return None, None

        item, id = in_memory_work_queue.pop(0)

    return item, id

async def get_work_item_no_wait(work_queue):
    global in_memory_work_queue
    global lock

    async with lock:
        if not in_memory_work_queue:
            return None, None

        item, id = in_memory_work_queue.pop(0)

    return item, id


async def fill_work_queue(work_queue):
    logger.debug("Filling work queue")

    request, id = await work_queue.get()

    if request is None:
        logger.debug("Nothing in the work queue")
        return

    item_path = request["path"]

    async with acquire_file_lock(item_path):
        with open(item_path, "r") as f:
            requests = json.load(f)

        logger.debug(f"Loaded {len(requests)} requests from {item_path} to work queue")

        group_request_id = strip_request_id(item_path)

        global in_memory_work_queue
        in_memory_work_queue = [
            (request, make_id(group_request_id, index))
            for index, request in enumerate(requests)
        ]

def make_id(group_request_id, index):
    return f"{group_request_id}_{index}"

def strip_request_id(item_path):
    base_name = os.path.basename(item_path)
    request_id, _ = os.path.splitext(base_name)
    return request_id
