
from cray_infra.util.get_config import get_config

from fastapi import Request, HTTPException, status
from starlette.requests import ClientDisconnect

from streaming_form_data.targets import FileTarget, ValueTarget
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.validators import MaxSizeValidator
from streaming_form_data.validators import ValidationError

from typing import Dict

import os
import tempfile
import hashlib
import time
import json
import tarfile
import shutil

import logging

logger = logging.getLogger(__name__)

async def upload_training_data(request: Request):

    try:
        temp_filepath = get_temp_filepath()

        config = get_config()
        max_file_size = config["max_upload_file_size"]
        max_request_body_size = config["max_upload_file_size"] * 2

        body_validator = MaxBodySizeValidator(max_file_size)

        file = FileTarget(temp_filepath, validator=MaxSizeValidator(max_file_size))
        params = ValueTarget()
        parser = StreamingFormDataParser(headers=request.headers)
        parser.register("file", file)
        parser.register("params", params)

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

        train_args = json.loads(params.value)

        train_args["dataset_hash"] = file_hash.hexdigest()

        job_directory = get_job_directory(train_args)

        train_args["job_directory"] = job_directory

        os.makedirs(job_directory, exist_ok=True)

        final_dataset_filepath = os.path.join(
            job_directory, "dataset.jsonlines"
        )

        train_args["training_data_path"] = final_dataset_filepath

        # extract the dataset from the tarball
        with tarfile.open(temp_filepath, "r") as tar:
            tar.extractall(job_directory)

        # add the ml directory if it doesn't exist
        ml_directory = os.path.join(job_directory, "ml")

        if not os.path.exists(ml_directory):
            logger.info(f"Copying ml directory to {ml_directory}")
            shutil.copytree(
                os.path.join(config["training_job_directory"], "..", "ml"),
                ml_directory,
            )

        # delete the tarball
        os.remove(temp_filepath)

        final_filepath = final_dataset_filepath

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

    return final_filepath, train_args


def get_temp_filepath():
    # Get a random temp file path
    return os.path.join(tempfile.gettempdir(), "training_job.tar.gz")


def get_file_hash(filepath: str):
    # Get the hash of the file
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash


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

def get_job_directory(train_args: Dict):
    contents = json.dumps(train_args)
    hash_id = hashlib.sha256(contents.encode()).hexdigest()

    config = get_config()

    job_directory = os.path.join(config["training_job_directory"], hash_id)

    return job_directory

def get_file_size(filepath: str):
    try:
        return os.path.getsize(filepath)
    except FileNotFoundError:
        return 0

