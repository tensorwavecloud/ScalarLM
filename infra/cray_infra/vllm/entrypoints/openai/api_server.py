import asyncio
import importlib
import inspect
import multiprocessing
import os
import re
import signal
import socket
import tempfile
import traceback
import typing
import json
import sys
import copy
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from typing import AsyncIterator, Set

import uvloop
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import State
from starlette.routing import Mount
from typing_extensions import assert_never

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError

# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              CompletionResponse,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest,
                                              EmbeddingResponse, ErrorResponse,
                                              LoadLoraAdapterRequest,
                                              TokenizeRequest,
                                              TokenizeResponse,
                                              UnloadLoraAdapterRequest)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, get_open_zmq_ipc_path
from vllm.version import __version__ as VLLM_VERSION

from cray_infra.util.get_config import get_config
from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session


TIMEOUT_KEEP_ALIVE = 5  # seconds

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger("vllm.entrypoints.openai.api_server")

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:

        # Get work task
        get_work_task = asyncio.create_task(get_work(app))
        _running_tasks.add(get_work_task)
        get_work_task.add_done_callback(_running_tasks.remove)

        # Log stats task
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(10.0)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log())
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove)
        else:
            task = None

        try:
            yield
        finally:
            if task is not None:
                task.cancel()
            get_work_task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


async def get_work(app: FastAPI):
    while True:
        try:
            await get_work_step(app)
        except AsyncEngineDeadError:
            logger.error("Engine is dead, restarting")
            # Kill the container so it can be restarted
            kill_vllm_container()
        except MQEngineDeadError:
            logger.error("Engine is dead, restarting")
            # Kill the container so it can be restarted
            kill_vllm_container()

        except Exception as e:
            logger.error("Error in get_work_step: %s", e)
            logger.error(traceback.format_exc())
            await asyncio.sleep(10)


def kill_vllm_container():
    # Kill instances of pt_thread_main process
    os.system("pgrep pt_main_thread | xargs kill -9")
    os.system("pgrep python | xargs kill -9")
    sys.exit(1)


async def get_work_step(app: FastAPI):

    try:
        await app.state.engine_client.check_health()
    except Exception as e:
        logger.error("Error in get_work_step: %s", e)
        raise AsyncEngineDeadError

    config = get_config()

    batch_size = await app.state.engine_client.get_dynamic_batch_size()
    max_batch_size = config["generate_batch_size"]

    params = {
        "batch_size": min(batch_size, max_batch_size),
    }

    logger.info("Getting work with params: %s", params)

    session = get_global_session()
    async with session.post(
        config["api_url"] + "/v1/generate/get_work",
        json=params,
    ) as resp:
        assert resp.status == 200

        data = await resp.json()

    if len(data["requests"]) == 0:
        logger.info("No work to do")
        return

    logger.info("Got work: %s", truncate_fields(data))

    completion_tasks = [
        async_generate_task(request, app) for request in data["requests"]
    ]

    results = await asyncio.gather(*completion_tasks)

    params = {
        "requests": results,
    }

    logger.info("Sending finished inference results with params: %s", params)

    async with session.post(
        config["api_url"] + "/v1/generate/finish_work",
        json=params,
    ) as resp:
        assert resp.status == 200


def truncate_fields(data):
    # Limit the length of the data to 100 characters
    # Data is a dict with a field called requests which is a list of dicts

    data = copy.deepcopy(data)

    for request in data["requests"]:
        for key, value in request.items():
            if isinstance(value, str) and len(value) > 100:
                request[key] = value[:100] + "..."
    return data


async def pass_receive() -> typing.NoReturn:
    return {"type": "http.request"}


async def async_generate_task(request, app):
    if request["request_type"] == "generate":
        return await async_completion_task(request, app)
    elif request["request_type"] == "embed":
        return await async_embedding_task(request, app)
    else:
        raise ValueError(f"Invalid request type: {request['request_type']}")


async def async_completion_task(request, app):
    completion_request = CompletionRequest(
        model=request["model"],
        prompt=request["prompt"],
        max_tokens=request["max_tokens"],
        temperature=0.0,
    )

    raw_request = Request(
        scope={"app": app, "type": "http", "headers": {}, "path": "/v1/completions"},
        receive=pass_receive,
    )

    response = await create_completion(completion_request, raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {
        "request_id": request["request_id"],
    }

    if "choices" in response_data:
        response["response"] = response_data["choices"][0]["text"]
    elif response_data["object"] == "error":
        response["error"] = response_data["message"]

    if "usage" in response_data:
        response["token_count"] = response_data["usage"]["total_tokens"]
        response["flop_count"] = (
            compute_flop_count(await app.state.engine_client.get_model_config())
            * response_data["usage"]["total_tokens"]
        )

    app.state.engine_client.check_health()

    return response


def compute_flop_count(model_config):

    # The intermediate size is the size of the feedforward layer
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    head_size = model_config.get_head_size()

    num_layers = getattr(model_config.hf_text_config, "num_hidden_layers", 12)
    num_attention_heads = getattr(
        model_config.hf_text_config, "num_attention_heads", 12
    )
    num_kv_heads = model_config.get_total_num_kv_heads()

    intermediate_size = getattr(
        model_config.hf_text_config, "intermediate_size", 4 * hidden_size
    )

    q_proj_flops = hidden_size * (num_attention_heads * head_size)
    kv_proj_flops = hidden_size * (num_kv_heads * head_size * 2)  # K and V

    # Attention computation: Q @ K^T and Attention @ V
    # Q @ K^T: [batch, heads, seq_len, head_size] @ [batch, heads, head_size, kv_cache_len]
    qk_flops = num_attention_heads * head_size

    # Attention @ V: [batch, heads, seq_len, kv_cache_len] @ [batch, heads, kv_cache_len, head_size]
    av_flops = num_attention_heads * head_size

    # Output projection: [batch, seq_len, hidden] @ [hidden, hidden]
    o_proj_flops = hidden_size * hidden_size

    attention_flops_per_layer = (
        q_proj_flops + kv_proj_flops + qk_flops + av_flops + o_proj_flops
    )
    total_attention_flops = attention_flops_per_layer * num_layers

    fc1_flops = hidden_size * intermediate_size
    fc2_flops = intermediate_size * hidden_size
    mlp_flops_per_layer = fc1_flops + fc2_flops

    total_mlp_flops = mlp_flops_per_layer * num_layers

    output_projection_flops = hidden_size * vocab_size

    embedding_flops = hidden_size * vocab_size

    total_flops = (
        total_attention_flops
        + total_mlp_flops
        + embedding_flops
        + output_projection_flops
    )

    return total_flops


async def async_embedding_task(request, app):
    embedding_request = EmbeddingRequest(
        model=request["model"],
        input=request["prompt"],
    )

    raw_request = Request(
        scope={"app": app, "type": "http", "headers": {}, "path": "/v1/embeddings"},
        receive=pass_receive,
    )

    response = await create_embedding(embedding_request, raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {
        "request_id": request["request_id"],
    }

    if "data" in response_data:
        if "embedding" in response_data["data"][0]:
            response["response"] = response_data["data"][0]["embedding"]
    elif response_data["object"] == "error":
        response["error"] = response_data["message"]

    return response


@asynccontextmanager
async def build_async_engine_client(args: Namespace) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    engine_args = AsyncEngineArgs.from_cli_args(args)

    async with build_async_engine_client_from_engine_args(
        engine_args, args.disable_frontend_multiprocessing
    ) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Fall back
    # TODO: fill out feature matrix.
    if (
        MQLLMEngineClient.is_unsupported_config(engine_args)
        or disable_frontend_multiprocessing
    ):
        engine_config = engine_args.create_engine_config()
        uses_ray = getattr(
            AsyncLLMEngine._get_executor_cls(engine_config), "uses_ray", False
        )

        build_engine = partial(
            AsyncLLMEngine.from_engine_args,
            engine_args=engine_args,
            engine_config=engine_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        if uses_ray:
            # Must run in main thread with ray for its signal handlers to work
            engine_client = build_engine()
        else:
            engine_client = await asyncio.get_running_loop().run_in_executor(
                None, build_engine
            )

        yield engine_client
        return

    # Otherwise, use the multiprocessing AsyncLLMEngine.
    else:
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            # Make TemporaryDirectory for prometheus multiprocessing
            # Note: global TemporaryDirectory will be automatically
            #   cleaned up upon exit.
            global prometheus_multiproc_dir
            prometheus_multiproc_dir = tempfile.TemporaryDirectory()
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
        else:
            logger.warning(
                "Found PROMETHEUS_MULTIPROC_DIR was set by user. "
                "This directory must be wiped between vLLM runs or "
                "you will find inaccurate metrics. Unset the variable "
                "and vLLM will properly handle cleanup."
            )

        # Select random path for IPC.
        ipc_path = get_open_zmq_ipc_path()
        logger.info("Multiprocessing frontend to use %s for IPC Path.", ipc_path)

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        context = multiprocessing.get_context("spawn")

        engine_process = context.Process(
            target=run_mp_engine,
            args=(engine_args, UsageContext.OPENAI_API_SERVER, ipc_path),
        )
        engine_process.start()
        logger.info("Started engine process with PID %d", engine_process.pid)

        # Build RPCClient, which conforms to EngineClient Protocol.
        # NOTE: Actually, this is not true yet. We still need to support
        # embedding models via RPC (see TODO above)
        engine_config = engine_args.create_engine_config()
        mp_engine_client = MQLLMEngineClient(ipc_path, engine_config)

        try:
            while True:
                try:
                    await mp_engine_client.setup()
                    break
                except TimeoutError:
                    if not engine_process.is_alive():
                        raise RuntimeError("Engine process failed to start") from None

            yield mp_engine_client  # type: ignore[misc]
        finally:
            # Ensure rpc server process was terminated
            engine_process.terminate()

            # Close all open connections to the backend
            mp_engine_client.close()

            # Wait for engine process to join
            engine_process.join(4)
            if engine_process.exitcode is None:
                # Kill if taking longer than 5 seconds to stop
                engine_process.kill()

            # Lazy import for prometheus multiprocessing.
            # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
            # before prometheus_client is imported.
            # See https://prometheus.github.io/client_python/multiprocess/
            from prometheus_client import multiprocess

            multiprocess.mark_process_dead(engine_process.pid)


router = APIRouter()


def mount_metrics(app: FastAPI):
    # Lazy import for prometheus multiprocessing.
    # We need to set PROMETHEUS_MULTIPROC_DIR environment variable
    # before prometheus_client is imported.
    # See https://prometheus.github.io/client_python/multiprocess/
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
    if prometheus_multiproc_dir_path is not None:
        logger.info(
            "vLLM to use %s as PROMETHEUS_MULTIPROC_DIR", prometheus_multiproc_dir_path
        )
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app(registry=registry))
    else:
        # Add prometheus asgi middleware to route /metrics requests
        metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


def chat(request: Request) -> OpenAIServingChat:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> OpenAIServingCompletion:
    return request.app.state.openai_serving_completion


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def embedding(request: Request) -> OpenAIServingEmbedding:
    return request.app.state.openai_serving_embedding


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """Health check."""
    await engine_client(raw_request).check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, TokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest, raw_request: Request):
    generator = await tokenization(raw_request).create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, DetokenizeResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    models = await completion(raw_request).show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    logger.info(f"Received request: {request.dict()}")
    logger.info(f"Received raw request: {raw_request.json()}")

    generator = await chat(raw_request).create_chat_completion(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await completion(raw_request).create_completion(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await embedding(raw_request).create_embedding(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    elif isinstance(generator, EmbeddingResponse):
        return JSONResponse(content=generator.model_dump())

    assert_never(generator)


if envs.VLLM_TORCH_PROFILER_DIR:
    logger.warning(
        "Torch Profiler is enabled in the API server. This should ONLY be "
        "used for local development!"
    )

    @router.post("/start_profile")
    async def start_profile(raw_request: Request):
        logger.info("Starting profiler...")
        await engine_client(raw_request).start_profile()
        logger.info("Profiler started.")
        return Response(status_code=200)

    @router.post("/stop_profile")
    async def stop_profile(raw_request: Request):
        logger.info("Stopping profiler...")
        await engine_client(raw_request).stop_profile()
        logger.info("Profiler stopped.")
        return Response(status_code=200)


@router.post("/v1/load_lora_adapter")
async def load_lora_adapter(request: LoadLoraAdapterRequest, raw_request: Request):
    logger.info(f"Received request: {request.dict()}")
    response = await chat(raw_request).load_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.code)

    response = await completion(raw_request).load_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.code)

    return Response(status_code=200, content=response)


@router.post("/v1/unload_lora_adapter")
async def unload_lora_adapter(request: UnloadLoraAdapterRequest, raw_request: Request):
    response = await chat(raw_request).unload_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.code)

    response = await completion(raw_request).unload_lora_adapter(request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.code)

    return Response(status_code=200, content=response)


def build_app(args: Namespace) -> FastAPI:
    logger.info("Building VLLM API server app")
    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    else:
        app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        chat = app.state.openai_serving_chat
        err = chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. " f"Must be a function or a class."
            )

    return app


def init_app_state(
    engine_client: EngineClient,
    model_config: ModelConfig,
    state: State,
    args: Namespace,
) -> None:
    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats

    state.openai_serving_chat = OpenAIServingChat(
        engine_client,
        model_config,
        base_model_paths,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        tool_parser=args.tool_call_parser,
    )
    state.openai_serving_completion = OpenAIServingCompletion(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
    )
    state.openai_serving_embedding = OpenAIServingEmbedding(
        engine_client,
        model_config,
        base_model_paths,
        request_logger=request_logger,
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        model_config,
        base_model_paths,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )


async def run_server(args, running_status, **uvicorn_kwargs) -> None:

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valide_tool_parses)} }})"
        )

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            # fd=sock.fileno(),
            running_status=running_status,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
