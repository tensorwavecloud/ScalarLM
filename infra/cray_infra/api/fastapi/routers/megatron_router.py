from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse,
)
from cray_infra.util.get_config import get_config

from cray_infra.training.launch_training_job import launch_training_job

from fastapi import APIRouter, Request, HTTPException, status
from starlette.requests import ClientDisconnect

import streaming_form_data
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

import logging

logger = logging.getLogger(__name__)

megatron_router = APIRouter(prefix="/megatron")


@megatron_router.post("/train")
async def train(request: Request):
    logger.info(f"Training request received: {request}")
    training_data_path, params = await upload_training_data(request)

    try:
        train_args = json.loads(params)

        logger.info(f"Training args: {train_args}")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid request body",
        )

    train_args["training_data_path"] = training_data_path

    job_info = await launch_training_job(train_args)

    return TrainResponse(
        job_id=job_info["job_id"],
        status=job_info["status"],
        message="Training job launched",
        dataset_id=os.path.basename(training_data_path).split(".")[0],
        job_directory=job_info["job_directory"],
        model_name=job_info["model_name"],
    )


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
        logger.info(f"Uploaded file size: {os.path.getsize(temp_filepath)} bytes")
        logger.info(
            f"Transfer rate: {os.path.getsize(temp_filepath) / (1.0e6 * (end_time - start_time))} MB/sec"
        )

        file_hash = get_file_hash(temp_filepath)

        train_args = params.value

        job_directory = get_job_directory(train_args)

        train_args["job_directory"] = job_directory

        os.makedirs(job_directory, exist_ok=True)

        final_filepath = os.path.join(
            job_directory, "dataset_" + file_hash.hexdigest() + ".jsonlines"
        )

        os.rename(temp_filepath, final_filepath)

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
    return os.path.join(tempfile.gettempdir(), "training_data.jsonlines")


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
