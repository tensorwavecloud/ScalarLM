from cray_infra.api.work_queue.push_into_queue import push_into_queue

from cray_infra.util.get_config import get_config

from fastapi import Request, HTTPException, status
from starlette.requests import ClientDisconnect

from streaming_form_data.targets import FileTarget
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.validators import MaxSizeValidator
from streaming_form_data.validators import ValidationError

from typing import Dict

import os
import tempfile
import hashlib
import time
import uuid
import shutil

import logging

logger = logging.getLogger(__name__)


async def upload(request: Request):
    try:
        temp_filepath = get_temp_filepath()

        config = get_config()
        max_file_size = config["max_upload_file_size"]
        max_request_body_size = config["max_upload_file_size"] * 2

        body_validator = MaxBodySizeValidator(max_file_size)

        file = FileTarget(temp_filepath, validator=MaxSizeValidator(max_file_size))
        parser = StreamingFormDataParser(headers=request.headers)
        parser.register("file", file)

        start_time = time.time()
        async for chunk in request.stream():
            body_validator(chunk)
            parser.data_received(chunk)

        end_time = time.time()

        logger.info(f"Uploaded file in {end_time - start_time} seconds")
        logger.info(f"Uploaded file to {temp_filepath}")
        logger.info(f"Uploaded file size: {get_file_size(temp_filepath)} bytes")
        logger.info(
            f"Transfer rate: {get_file_size(temp_filepath) / (1.0e6 * (end_time - start_time))} MB/sec"
        )

        file_hash = get_file_hash(temp_filepath)

        request_path = get_request_path(file_hash.hexdigest())

        # rename the file to its final destination
        os.makedirs(os.path.dirname(request_path), exist_ok=True)
        shutil.move(temp_filepath, request_path)

        await push_into_queue(request_path)

    except ClientDisconnect:
        logger.warning("Client Disconnected")
    except MaxBodySizeException as e:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Maximum request body size limit ({max_request_body_size} bytes) exceeded ({e.body_len} bytes read)",
        )
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Maximum file size limit ({max_file_size} bytes) exceeded",
        )
    except Exception as e:
        # Log the backtrace
        logger.exception(e)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"There was an error uploading the file: {e}",
        )

    if not file.multipart_filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="File is missing"
        )


def get_temp_filepath() -> str:
    temp_dir = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())
    temp_filepath = os.path.join(temp_dir, f"upload_{unique_id}.json")
    return temp_filepath


def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0


def get_file_hash(filepath: str) -> hashlib._hashlib.HASH:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            data = f.read(65536)  # Read in 64k chunks
            if not data:
                break
            sha256.update(data)

    return sha256


def get_request_path(file_hash: str) -> str:
    config = get_config()
    base_path = config["upload_base_path"]
    os.makedirs(base_path, exist_ok=True)
    request_path = os.path.join(base_path, f"{file_hash}.json")
    return request_path


class MaxBodySizeException(Exception):
    def __init__(self, body_len: str):
        self.body_len = body_len


class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=self.body_len)
