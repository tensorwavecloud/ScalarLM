# Set vLLM environment variables BEFORE any vLLM imports
import os
import torch


from cray_infra.util.get_config import get_config
from cray_infra.huggingface.get_hf_token import get_hf_token

from vllm.entrypoints.openai.api_server import build_app, decorate_logs, \
    init_app_state, maybe_register_tokenizer_info_endpoint, setup_server, \
    load_log_config, build_async_engine_client


from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.utils import log_non_default_args
import vllm.envs as envs

import uvicorn
import logging

logger = logging.getLogger(__name__)

def set_cpu():
    # Set device target before vLLM imports for proper device inference
    print("No CUDA available, forcing CPU platform")
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Enable debug logging as suggested by error
    # Set additional vLLM CPU environment variables
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_MODELSCOPE"] = "False"
    os.environ["VLLM_USE_V1"] = "1"

    # Remove CUDA_VISIBLE_DEVICES for CPU mode to avoid device conflicts
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

async def create_vllm(server_status, port):
    print(f"DEBUG: BEFORE CONFIG - Environment variables:")
    print(f"  VLLM_TARGET_DEVICE: {os.environ.get('VLLM_TARGET_DEVICE', 'NOT SET')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

    os.environ["HUGGING_FACE_HUB_TOKEN"] = get_hf_token()

    config = get_config()

    if config['dtype'] == 'auto':
        # Set to float32 on the cpu
        if not torch.cuda.is_available():
            config['dtype'] = 'float32'

    # Set backend to FLASHMLA on cuda sm version less than 8.0
    if torch.cuda.is_available():
        sm_version = torch.cuda.get_device_capability()[0]
        if sm_version < 8:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHMLA"
            config['dtype'] = 'float32'
            os.environ["VLLM_USE_STANDALONE_COMPILE"] = "0"
            print(f"DEBUG: Setting VLLM_BACKEND=flashmla for sm_version {sm_version}")
        else:
            print(f"DEBUG: Using default VLLM_BACKEND for sm_version {sm_version}")


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
    ]


    if config['limit_mm_per_prompt'] is not None:
        args.append(f"--limit-mm-per-prompt={config['limit_mm_per_prompt']}")

    if torch.cuda.is_available():
        args.append("--device=cuda")
    #else:
    #    set_cpu()


    args = parser.parse_args(args=args)

    args.port = port
    args.model = config["model"]

    logger.info(f"Running vLLM with args: {args}")

    await run_server(server_status, args)

async def run_server(server_status, args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer")

    listen_address, sock = setup_server(args)
    await run_server_worker(server_status, listen_address, sock, args, **uvicorn_kwargs)

async def run_server_worker(server_status, listen_address,
                            sock,
                            args,
                            client_config=None,
                            **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs['log_config'] = log_config

    async with build_async_engine_client(
            args,
            client_config=client_config,
    ) as engine_client:

        logger.info("DEBUG: build_async_engine_client completed, engine_client created")
        
        maybe_register_tokenizer_info_endpoint(args)
        logger.info("DEBUG: maybe_register_tokenizer_info_endpoint completed")
        
        app = build_app(args)
        logger.info("DEBUG: build_app completed")
        
        server_status.set_app(app)
        logger.info("DEBUG: server_status.set_app completed - generate worker can now access app")

        logger.info("DEBUG: About to call engine_client.get_vllm_config() - THIS IS WHERE IT MAY HANG")
        vllm_config = await engine_client.get_vllm_config()
        logger.info("DEBUG: engine_client.get_vllm_config() completed successfully!")
        
        logger.info("DEBUG: About to call init_app_state")
        await init_app_state(engine_client, vllm_config, app.state, args)
        logger.info("DEBUG: init_app_state completed")

        logger.info("Starting vLLM API server %d on %s", server_index,
                    listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()
