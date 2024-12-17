from cray_infra.util.get_config import get_config

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

import uvicorn
import os

import logging

logger = logging.getLogger(__name__)


async def create_vllm(port, running_status):

    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=["--enable-lora"])

    config = get_config()

    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(args)
