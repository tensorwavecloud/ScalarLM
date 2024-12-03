from masint.util.make_api_url import make_api_url

import aiofiles
import aiohttp
import contextlib
import asyncio
import io
import json

import tempfile
import jsonlines


class AsyncCray:
    async def train(self, data, model_name, train_args):
        with make_data_file(data) as data_file_path:
            api_url = make_api_url("v1/megatron/train")

            return await upload_async(data_file_path, api_url, train_args)

    async def health(self):
        api_url = make_api_url("v1/health")
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def generate(self, prompts, model_name, max_tokens):
        api_url = make_api_url("v1/generate")
        async with aiohttp.ClientSession() as session:
            params = {"prompts" : prompts }

            if model_name is not None:
                params["model"] = model_name

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            async with session.post(api_url, json=params) as resp:
                return await resp.json()


async def upload_async(data_file_path, api_url, train_args):
    async with aiohttp.ClientSession() as session:

        with aiohttp.MultipartWriter("form-data") as mp:
            file_part = mp.append(file_sender(data_file_path))
            file_part.set_content_disposition(
                "form-data", name="file", filename="dataset"
            )

            params_part = mp.append_json(train_args)
            params_part.set_content_disposition("form-data", name="params")

            async with session.post(api_url, data=mp, headers=mp.headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to upload data: {await resp.text()}")
                return await resp.json()


async def file_sender(file_path):
    chunk_size = 64 * 1024  # 64 KB

    async with aiofiles.open(file_path, "rb") as f:
        chunk = await f.read(chunk_size)
        while chunk:
            yield chunk
            chunk = await f.read(chunk_size)


@contextlib.contextmanager
def make_data_file(data):
    chunk_size = 64 * 1024

    if isinstance(data, str):
        with open(data, "rb") as f:
            yield f

    elif isinstance(data, io.BufferedIOBase):
        with tempfile.NamedTemporaryFile() as f:
            for chunk in iter(lambda: data.read(chunk_size), b""):
                f.write(chunk)
            f.seek(0)
            yield f.name

    elif isinstance(data, list):
        with tempfile.NamedTemporaryFile() as f:
            with jsonlines.open(f.name, mode="w") as writer:
                for item in data:
                    writer.write(item)

            yield f.name

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
