from cray_infra.util.get_config import get_config
from cray_infra.huggingface.get_hf_token import get_hf_token

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser

import torch

import uvicorn
import os

import logging

logger = logging.getLogger(__name__)

async def create_vllm(port, running_status):

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    config = get_config()

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = [
        f"--dtype={config['dtype']}",
        f"--max-model-len={config['max_model_length']}",
        f"--max-num-batched-tokens={config['max_model_length']}",
        f"--max-seq-len-to-capture={config['max_model_length']}",
        f"--gpu-memory-utilization={config['gpu_memory_utilization']}",
        f"--max-log-len={config['max_log_length']}",
        f"--swap-space=0",
        "--enable-lora",
        "--disable-async-output-proc", # Disable async output processing for embeddings
    ]

    if config['limit_mm_per_prompt'] is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")

    if torch.cuda.is_available():
        args.append("--device=cuda")

    args = parser.parse_args(args=args)

    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(args, running_status)
