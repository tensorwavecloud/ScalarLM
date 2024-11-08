
from vllm.entrypoints.openai.api_server import run_server
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

import uvicorn

import logging

logger = logging.getLogger(__name__)

async def create_vllm(port, running_status):
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    args.port = port
    args.model = "lamini/tiny-random-llama"

    logger.info(f"Running vLLM with args: {args}")

    await run_server(args)

