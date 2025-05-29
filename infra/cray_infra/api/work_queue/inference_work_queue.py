
from cray_infra.util.get_config import get_config

import persistqueue


import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class InferenceWorkQueue:
    def __init__(self, path):
        self.queue = persistqueue.SQLiteAckQueue(path)

    def put(self, request):
        return self.queue.put(request)

    async def get(self):
        config = get_config()

        timeout = config["inference_work_queue_timeout"]

        start_time = time.time()

        item = None
        request_id = None

        while time.time() - start_time < timeout:
            try:
                raw_item = self.queue.get(block=False, raw=True)

                item = raw_item["data"]
                request_id = raw_item["pqid"]

                break

            except persistqueue.Empty:
                await asyncio.sleep(0.01)

        return item, request_id

    def get_id(self, id):
        return self.queue.get(block=False, id=id)

    def get_nowait(self):
        try:
            raw_item = self.queue.get(block=False, raw=True)

            item = raw_item["data"]
            request_id = raw_item["pqid"]
        except persistqueue.Empty:
            item = None
            request_id = None

        return item, request_id

    def get_if_finished(self, id):
        results = self.queue.queue()

        for result in results:
            if result["id"] == id:
                if int(result["status"]) == int(persistqueue.AckStatus.acked):
                    return result["data"]

        return None

    def update(self, id, item):
        return self.queue.update(id=id, item=item)

    def ack(self, id):
        return self.queue.ack(id=id)

    def resume_unack_tasks(self):
        self.queue.resume_unack_tasks()

    def clear_acked_data(self):
        self.queue.clear_acked_data()

    def __len__(self):
        return len(self.queue)

inference_work_queue = None

def get_inference_work_queue():
    global inference_work_queue

    if inference_work_queue is None:
        inference_work_queue = get_file_backed_inference_work_queue()

    return inference_work_queue


def get_file_backed_inference_work_queue():
    config = get_config()
    path = config["inference_work_queue_path"]

    return InferenceWorkQueue(path=path)

