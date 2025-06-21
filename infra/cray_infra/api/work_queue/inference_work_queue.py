
from cray_infra.util.get_config import get_config

import persistqueue


import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class InferenceWorkQueue:
    def __init__(self, path, auto_resume=False):
        self.queue = persistqueue.SQLiteAckQueue(path, auto_resume=auto_resume)
        self.lock = asyncio.Lock()

    async def put(self, request):
        async with self.lock:
            return self.queue.put(request)

    async def get(self):
        config = get_config()

        timeout = config["inference_work_queue_timeout"]

        start_time = time.time()

        item = None
        request_id = None

        while time.time() - start_time < timeout:
            try:
                async with self.lock:
                    raw_item = self.queue.get(block=False, raw=True)

                item = raw_item["data"]
                request_id = raw_item["pqid"]

                break

            except persistqueue.Empty:
                await asyncio.sleep(0.01)

        return item, request_id

    async def get_id(self, id):
        async with self.lock:
            return self.queue.get(block=False, id=id)

    async def get_nowait(self):
        try:
            async with self.lock:
                raw_item = self.queue.get(block=False, raw=True)

            item = raw_item["data"]
            request_id = raw_item["pqid"]

        except persistqueue.Empty:
            item = None
            request_id = None

        return item, request_id

    async def get_if_finished(self, id):
        async with self.lock:
            results = self.queue.queue()

            for result in results:
                if result["id"] == id:
                    if int(result["status"]) == int(persistqueue.AckStatus.acked):
                        return result["data"]

        return None

    async def update(self, id, item):
        async with self.lock:
            return self.queue.update(id=id, item=item)

    async def ack(self, id):
        async with self.lock:
            return self.queue.ack(id=id)

    async def update_and_ack(self, id, item):
        async with self.lock:
            result = self.queue.update(id=id, item=item)
            ack_result = self.queue.ack(id=id)

            return result, ack_result

    async def resume_unack_tasks(self):
        async with self.lock:
            self.queue.resume_unack_tasks()

    async def clear_acked_data(self):
        async with self.lock:
            self.queue.clear_acked_data()

    def __len__(self):
        return len(self.queue)

inference_work_queue = None
lock = asyncio.Lock()

async def get_inference_work_queue():
    global inference_work_queue
    global lock

    async with lock:
        if inference_work_queue is None:
            inference_work_queue = get_file_backed_inference_work_queue(auto_resume=True)

    return inference_work_queue


def get_file_backed_inference_work_queue(auto_resume=False):
    config = get_config()
    path = config["inference_work_queue_path"]

    return InferenceWorkQueue(path=path, auto_resume=auto_resume)

