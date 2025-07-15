from masint.util.make_api_url import make_api_url
from masint.engines.cray.submit_training_job import upload_async, tar_info_strip_file_info, make_data_file

import tempfile
import tarfile
import contextlib
import os

import logging

logger = logging.getLogger(__name__)

async def submit_slurm_job(code, train_args, api_url):
    with make_slurm_archive(code) as archive_path:
        api_url = make_api_url("v1/megatron/train", api_url=api_url)

        return await upload_async(archive_path, api_url, train_args)

@contextlib.contextmanager
def make_slurm_archive(code):

    data = [{"input": "example input", "output": "example output"}]

    with make_data_file(data) as data_file_path:
        with tempfile.NamedTemporaryFile() as archive_file:
            with tarfile.open(archive_file.name, "w") as tar:
                tar.add(
                    data_file_path,
                    arcname="dataset.jsonlines",
                    filter=tar_info_strip_file_info,
                )

                with make_ml_dir(code) as ml_dir:
                    tar.add(ml_dir, arcname="ml", filter=tar_info_strip_file_info)

            archive_file.seek(0)
            archive_file.flush()

            logger.debug(f"Archive created at {archive_file.name}")
            logger.debug(f"Archive size: {os.path.getsize(archive_file.name)} bytes")

            yield archive_file.name

@contextlib.contextmanager
def make_ml_dir(code):
    """
    Create a temporary directory for the machine learning code.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        ml_dir = os.path.join(temp_dir, "ml", "cray_megatron")
        os.makedirs(ml_dir, exist_ok=True)

        # Write the code to a file in the ml directory
        code_file_path = os.path.join(ml_dir, "main.py")
        with open(code_file_path, "w") as code_file:
            code_file.write(code)

        yield os.path.dirname(ml_dir)
