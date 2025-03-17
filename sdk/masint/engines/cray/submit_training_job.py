from masint.util.make_api_url import make_api_url

import aiofiles
import aiohttp
import contextlib
import asyncio
import io
import json
import os
import tarfile
import tempfile
import jsonlines


async def submit_training_job(data, model_name, train_args):
    with make_training_archive(data) as archive_path:
        api_url = make_api_url("v1/megatron/train")

        return await upload_async(archive_path, api_url, train_args)


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
def make_training_archive(data):

    with make_data_file(data) as data_file_path:
        with tempfile.NamedTemporaryFile() as archive_file:
            with tarfile.open(archive_file.name, "w") as tar:
                # Add the data file to the archive
                tar.add(data_file_path, arcname="dataset.jsonlines", filter=tar_info_strip_file_info)

                # Add the machine learning directory to the archive
                # The directory tree is as follows:
                #  - cray/sdk/masint/engines/cray/submit_training_job.py <- this file
                #  - cray/ml <- the machine learning directory
                ml_dir = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "..", "ml"
                )
                tar.add(ml_dir, arcname="ml", filter=tar_info_strip_file_info)

            archive_file.seek(0)
            yield archive_file.name

def tar_info_strip_file_info(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    tarinfo.mtime = 0
    return tarinfo


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
        with tempfile.NamedTemporaryFile(delete=False) as f:
            with jsonlines.open(f.name, mode="w") as writer:
                for item in data:
                    writer.write(item)

            yield f.name

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
