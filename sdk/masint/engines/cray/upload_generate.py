from masint.util.make_api_url import make_api_url

import aiohttp
import aiofiles
import contextlib
import tempfile
import json

import logging

logger = logging.getLogger(__name__)

async def upload_generate(prompts, model_name, max_tokens, api_url=None):

    with make_upload_json_file(prompts, model_name, max_tokens) as upload_path:

        api_url = make_api_url("v1/generate/upload", api_url=api_url)

        return await upload_async(upload_path, api_url)

@contextlib.contextmanager
def make_upload_json_file(prompts, model_name, max_tokens):
    requests_object = {
        "prompts": prompts,
        "model_name": model_name,
        "max_tokens": max_tokens
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        json.dump(requests_object, f)
        f.flush()
        f.seek(0)

        logger.debug(f"Created temporary upload file at {f.name}")
        logger.debug(f"Upload file size: {f.tell()} bytes")

        yield f.name


async def upload_async(data_file_path, api_url):
    async with aiohttp.ClientSession() as session:

        content_length = await get_content_length(data_file_path)

        with make_multipart_writer(data_file_path) as mp:

            headers = mp.headers

            headers["Content-Length"] = str(content_length)

            async with session.post(api_url, data=mp, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to upload data: {await resp.text()}")
                return await resp.json()

async def get_content_length(data_file_path):
    with make_multipart_writer(data_file_path) as mp:

        class Writer:
            def __init__(self):
                self.count = 0

            async def write(self, data):
                self.count += len(data)

        writer = Writer()
        await mp.write(writer)
        content_length = writer.count

        return content_length

@contextlib.contextmanager
def make_multipart_writer(data_file_path):
    with aiohttp.MultipartWriter("form-data") as mp:
        part = mp.append(file_sender(data_file_path))
        part.set_content_disposition("form-data", name="file", filename="requests.json")

    yield mp

async def file_sender(file_path):
    chunk_size = 64 * 1024

    async with aiofiles.open(file_path, "rb") as f:
        chunk = await f.read(chunk_size)
        index = 0
        while chunk:
            yield chunk
            chunk = await f.read(chunk_size)
            index += chunk_size
