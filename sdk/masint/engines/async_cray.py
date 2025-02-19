from masint.util.make_api_url import make_api_url

from masint.engines.cray.submit_training_job import submit_training_job

import aiohttp


class AsyncCray:
    async def train(self, data, model_name, train_args):
        return await submit_training_job(data, model_name, train_args)

    async def generate(self, prompts, model_name, max_tokens):
        result = await self.submit_generate(prompts, model_name, max_tokens)

        final_result = await poll_for_responses(result)

        return [response["response"] for response in final_result["results"]]

    async def submit_generate(self, prompts, model_name, max_tokens):
        api_url = make_api_url("v1/generate")
        async with aiohttp.ClientSession() as session:
            params = {"prompts": prompts}

            if model_name is not None:
                params["model"] = model_name

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            async with session.post(api_url, json=params) as resp:
                assert resp.status == 200
                return await resp.json()

    async def embed(self, prompts, model_name):
        result = await self.submit_embed(prompts, model_name)

        final_result = await poll_for_responses(result)

        return [response["response"] for response in final_result["results"]]

    async def submit_embed(self, prompts, model_name):
        api_url = make_api_url("v1/generate/embed")
        async with aiohttp.ClientSession() as session:
            params = {"prompts": prompts}

            if model_name is not None:
                params["model"] = model_name

            async with session.post(api_url, json=params) as resp:
                assert resp.status == 200
                return await resp.json()

    async def get_results(self, request_ids):
        async with aiohttp.ClientSession() as session:
            api_url = make_api_url("v1/generate/get_results")
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                return await resp.json()

    async def list_models(self):
        api_url = make_api_url("v1/megatron/list_models")
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def get_training_job(self, job_dir):
        api_url = make_api_url(f"v1/megatron/train/{job_dir}")
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def health(self):
        api_url = make_api_url("v1/health")
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()


async def poll_for_responses(result):
    api_url = make_api_url("v1/generate/get_results")

    async with aiohttp.ClientSession() as session:
        while not is_finished(result):
            request_ids = [response["request_id"] for response in result["results"]]
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                result = await resp.json()

    return result


def is_finished(result):
    for response in result["results"]:
        if response["error"] is not None:
            raise Exception(response["error"])

        if response["response"] is None:
            return False

    return True
