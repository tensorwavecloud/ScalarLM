from masint.util.make_api_url import make_api_url

from masint.engines.cray.submit_training_job import submit_training_job
from masint.engines.cray.submit_slurm_job import submit_slurm_job

from masint.engines.cray.upload_generate import upload_generate
from masint.engines.cray.poll_for_downloads import poll_for_downloads

import aiohttp

import logging

logger = logging.getLogger(__name__)


class AsyncCray:
    def __init__(self, api_url=None):
        self.api_url = api_url

    async def train(self, data, model_name, train_args):
        return await submit_training_job(data, model_name, train_args, api_url=self.api_url)

    async def submit_slurm_job(self, code, train_args=None):
        return await submit_slurm_job(code, train_args, api_url=self.api_url)

    async def generate(self, prompts, model_name, max_tokens):

        upload_threshold = 128

        if len(prompts) > upload_threshold:
            result = await upload_generate(prompts, model_name, max_tokens, api_url=self.api_url)

            handle_error(result)

            final_result = await poll_for_downloads(result, api_url=self.api_url)
        else:
            result = await self.submit_generate(prompts, model_name, max_tokens)

            handle_error(result)

            final_result = await poll_for_responses(result, api_url=self.api_url)

        return [response["response"] for response in final_result["results"]]

    async def submit_generate(self, prompts, model_name, max_tokens):
        api_url = make_api_url("v1/generate", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            params = {"prompts": prompts}

            if model_name is not None:
                params["model"] = model_name

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            async with session.post(api_url, json=params) as resp:
                assert resp.status == 200
                return await resp.json()

    async def get_results(self, request_ids):
        async with aiohttp.ClientSession() as session:
            api_url = make_api_url("v1/generate/get_results", api_url=self.api_url)
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                return await resp.json()

    async def list_models(self):
        api_url = make_api_url("v1/megatron/list_models", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def get_training_job(self, job_dir):
        api_url = make_api_url(f"v1/megatron/train/{job_dir}", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def health(self):
        api_url = make_api_url("v1/health", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def metrics(self):
        api_url = make_api_url("v1/generate/metrics", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def get_gpu_count(self):
        api_url = make_api_url("v1/megatron/gpu_count", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                response = await resp.json()
                logger.debug(f"get_gpu_count response: {response}")
                return response["gpu_count"]

    async def get_node_count(self):
        api_url = make_api_url("v1/megatron/node_count", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                response = await resp.json()
                logger.debug(f"get_node_count response: {response}")
                return response["node_count"]

    async def cancel(self, model_name):
        api_url = make_api_url(f"v1/megatron/cancel/{model_name}", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url) as resp:
                return await resp.json()

    async def clear_queue(self):
        api_url = make_api_url("v1/generate/clear_queue", api_url=self.api_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url) as resp:
                return await resp.json()


def handle_error(result):
    if "error" in result and result["error"] is not None:
        logger.error(f"Error in response: {result['error']}")
        raise Exception(result["error"])

    if not result.get("results"):
        logger.error(f"No results found in response: {result}")
        raise Exception("No results found in response")

    if not isinstance(result["results"], list):
        logger.error(f"Results is not a list: {result['results']}")
        raise Exception("Results is not a list")


async def poll_for_responses(result, api_url):
    api_url = make_api_url("v1/generate/get_results", api_url=api_url)

    async with aiohttp.ClientSession() as session:
        while not is_finished(result):
            request_ids = [response["request_id"] for response in result["results"]]
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                result = await resp.json()

            handle_error(result)

    return result


def is_finished(result):
    for response in result["results"]:
        if response["error"] is not None:
            raise Exception(response["error"])

        if response["response"] is None:
            return False

    return True
