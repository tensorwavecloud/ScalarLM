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

    config = get_config()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[
        f"--dtype={config['dtype']}",
        f"--max-model-len={config['max_model_length']}",
        f"--max-num-batched-tokens={config['max_model_length']}",
        f"--max-seq-len-to-capture={config['max_model_length']}",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        "--enable-lora",
        "--disable-async-output-proc", # Disable async output processing for embeddings
    ])


    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(args, running_status)
