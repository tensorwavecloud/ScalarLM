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

import logging

logger = logging.getLogger(__name__)


async def submit_training_job(data, model_name, train_args, api_url):
    with make_training_archive(data) as archive_path:
        api_url = make_api_url("v1/megatron/train", api_url=api_url)

        return await upload_async(archive_path, api_url, train_args)


async def upload_async(data_file_path, api_url, train_args):
    async with aiohttp.ClientSession() as session:

        content_length = await get_content_length(data_file_path, train_args)

        with make_multipart_writer(data_file_path, train_args) as mp:

            headers = mp.headers

            headers["Content-Length"] = str(content_length)

            async with session.post(api_url, data=mp, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to upload data: {await resp.text()}")
                return await resp.json()


async def get_content_length(data_file_path, train_args):
    with make_multipart_writer(data_file_path, train_args) as mp:

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
def make_multipart_writer(data_file_path, train_args):
    with aiohttp.MultipartWriter("form-data") as mp:
        file_part = mp.append(file_sender(data_file_path))
        file_part.set_content_disposition("form-data", name="file", filename="dataset")

        params_part = mp.append_json(train_args)
        params_part.set_content_disposition("form-data", name="params")

        yield mp


async def file_sender(file_path):
    chunk_size = 64 * 1024  # 64 KB

    async with aiofiles.open(file_path, "rb") as f:
        chunk = await f.read(chunk_size)
        index = 0
        while chunk:
            yield chunk
            chunk = await f.read(chunk_size)
            index += chunk_size


@contextlib.contextmanager
def make_training_archive(data):

    with make_data_file(data) as data_file_path:
        check_for_zero_length_file(data_file_path)

        with tempfile.NamedTemporaryFile() as archive_file:
            with tarfile.open(archive_file.name, "w") as tar:
                # Add the data file to the archive
                tar.add(
                    data_file_path,
                    arcname="dataset.jsonlines",
                    filter=tar_info_strip_file_info,
                )

                # Add the machine learning directory to the archive
                ml_dir = find_ml_dir()

                if ml_dir is None:
                    logger.warning(
                        f"ML directory not found. Skipping addition to "
                        "archive, using default ml directory from ScalarLM server."
                    )
                else:
                    tar.add(ml_dir, arcname="ml", filter=tar_info_strip_file_info)

            archive_file.seek(0)
            archive_file.flush()

            logger.debug(f"Archive created at {archive_file.name}")
            logger.debug(f"Archive size: {os.path.getsize(archive_file.name)} bytes")

            yield archive_file.name


def tar_info_strip_file_info(tarinfo):
    tarinfo.uid = tarinfo.gid = 0
    tarinfo.uname = tarinfo.gname = "root"
    tarinfo.mtime = 0
    return tarinfo


def find_ml_dir():
    # Check the current directory first
    current_directory = os.path.join(os.getcwd(), "ml")

    if os.path.exists(current_directory):
        return current_directory

    # The directory tree is as follows:
    #  - cray/sdk/masint/engines/cray/submit_training_job.py <- this file
    #  - cray/ml <- the machine learning directory
    peer_directory = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "ml"
    )

    if os.path.exists(peer_directory):
        return peer_directory


@contextlib.contextmanager
def make_data_file(data):

    if isinstance(data, str):
        with open(data, "rb") as f:
            yield f

    elif isinstance(data, io.BufferedIOBase):
        chunk_size = 64 * 1024

        with tempfile.NamedTemporaryFile() as f:
            for chunk in iter(lambda: data.read(chunk_size), b""):
                f.write(chunk)
            f.seek(0)
            f.flush()
            yield f.name

    elif isinstance(data, list):
        with tempfile.NamedTemporaryFile() as f:
            with jsonlines.open(f.name, mode="w") as writer:
                for item in data:
                    writer.write(item)

            yield f.name

    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def check_for_zero_length_file(file_path):
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"The file {file_path} is empty. Please provide valid data.")
