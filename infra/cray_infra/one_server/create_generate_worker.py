import asyncio
import aiohttp
import copy
import json
import os
import sys

import logging

from typing import Optional, NoReturn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.api_server import (
    load_lora_adapter,
    create_chat_completion,
    create_completion,
)

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    LoadLoRAAdapterRequest,
)

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


from cray_infra.api.fastapi.routers.request_types.get_work_response import (
    PromptType,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.util.get_config import get_config

logger = logging.getLogger(__name__)


async def create_generate_worker(server_status):
    """
    Create a worker that processes generate requests from the queue.
    Uses configurable vLLM engine (HTTP or direct) for processing.
    """
    logger.info("=" * 60)
    logger.info("STARTING GENERATE WORKER")
    logger.info("=" * 60)

    config = get_config()
    api_base = config["api_url"]

    app = await server_status.get_app()

    # Main worker loop
    session: Optional[aiohttp.ClientSession] = None

    tasks = []

    loaded_adaptor_count = 0

    try:
        session = aiohttp.ClientSession()

        while True:
            clear_finished_tasks(tasks)

            batch_size = await get_batch_size(app)

            if batch_size == 0:
                logger.debug(f"No batch size available, waiting for kv cache space")
                await asyncio.sleep(0.1)
                continue

            logger.debug(f"Checking for work with batch size: {batch_size}, loaded adaptors: {loaded_adaptor_count}...")

            try:
                get_work_response = await session.post(
                    f"{api_base}/v1/generate/get_work",
                    json={"batch_size": batch_size, "loaded_adaptor_count": loaded_adaptor_count},
                    timeout=aiohttp.ClientTimeout(total=config["inference_work_queue_timeout"]),
                )

            except asyncio.TimeoutError:
                logger.debug("Timeout waiting for work, retrying...")
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
                continue

            if get_work_response.status != 200:
                logger.error(f"Failed to get work: {get_work_response.status}")
                await asyncio.sleep(1)
                continue

            work_data = await get_work_response.json()

            adaptors = work_data["new_adaptors"]["new_adaptors"]

            loaded_adaptor_count = await add_adaptors(app, adaptors, loaded_adaptor_count)

            requests = work_data.get("requests", [])

            if not requests:
                logger.debug("No work available")
                await asyncio.sleep(1)
                continue

            new_task = await process_requests(app, requests)

            tasks.append(new_task)

    except Exception as e:
        # print backtrace for debugging
        logger.error(f"Worker encountered an error: {e}", exc_info=True)
        logger.error("Shutting down worker due to error")
        kill_vllm_container()

    finally:
        # Cleanup resources
        if session and not session.closed:
            await session.close()

        logger.info("Generate worker shutting down")


def clear_finished_tasks(tasks):
    # Handle exceptions
    for task in tasks:
        if task.done():
            try:
                task.result()  # Get the result to raise any exceptions
            except Exception as e:
                logger.error(f"Task raised an exception: {e}", exc_info=True)

    tasks[:] = [task for task in tasks if not task.done()]

    # Log the number of remaining tasks
    logger.debug(f"Remaining tasks: {len(tasks)}")


async def get_batch_size(app):
    vllm_engine_client = app.state.engine_client
    current_kv_cache_size = await vllm_engine_client.get_current_kv_cache_size()

    config = get_config()

    batch_size = current_kv_cache_size // config["max_model_length"]

    if batch_size <= 0:
        logger.debug("Batch size is 0, waiting for kv cache space")
        return 0

    current_kv_cache_size = await vllm_engine_client.get_current_kv_cache_size()

    batch_size = current_kv_cache_size // config["max_model_length"]

    return batch_size


async def add_adaptors(app, adaptors, loaded_adaptor_count):
    config = get_config()

    params = {
        "loaded_adaptor_count": loaded_adaptor_count,
    }

    try:
        session = get_global_session()

        for new_adaptor in adaptors:
            logger.info("Loading new adaptor: %s", new_adaptor)
            try:
                await add_new_adaptor(app, new_adaptor)
                loaded_adaptor_count += 1
            except Exception as e:
                logger.error("Error loading adaptor %s: %s", new_adaptor, e)
                continue
    except Exception as e:
        logger.error("Error loading adaptors %s", e)

    return loaded_adaptor_count


async def add_new_adaptor(app: FastAPI, new_adaptor: str):
    config = get_config()
    base_path = config["training_job_directory"]

    new_adaptor_path = os.path.join(base_path, new_adaptor)

    logger.info("Loading new adaptor from path: %s", new_adaptor_path)

    lora_adaptor_request = LoadLoRAAdapterRequest(lora_name=new_adaptor, lora_path=new_adaptor_path)

    raw_request = Request(
        scope={
            "app": app,
            "type": "http",
            "headers": {},
            "path": "/v1/load_lora_adapter",
        },
        receive=pass_receive,
    )

    response = await load_lora_adapter(lora_adaptor_request, raw_request=raw_request)

    if isinstance(response, JSONResponse):
        if response.status_code != 200:
            logger.error(
                "Failed to load new adaptor %s: %s",
                new_adaptor,
                response.content.decode("utf-8"),
            )
            raise RuntimeError(
                f"Failed to load new adaptor {new_adaptor}: {response.content.decode('utf-8')}"
            )
        else:
            logger.info("Successfully loaded new adaptor: %s", new_adaptor)


async def process_requests(app, requests):
    """
    Process a batch of requests using the vLLM engine.
    This function creates tasks for each request and returns them.
    """

    # Create a task to process the requests
    return asyncio.create_task(process_requests_task(app, requests))


async def process_requests_task(app, requests):
    logger.info(f"Processing {len(requests)} requests ")

    config = get_config()

    logger.info("Got work: %s", truncate_fields({"requests": requests}))

    completion_tasks = [async_generate_task(request, app) for request in requests]

    results = await asyncio.gather(*completion_tasks, return_exceptions=True)

    # Handle exceptions
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} raised an exception: {result}", exc_info=True)
            results[i] = {
                "request_id": requests[i]["request_id"],
                "error": str(result),
            }

    params = {"requests": results}

    logger.info("Sending finished inference results with params: %s", params)

    config = get_config()

    session = get_global_session()
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


async def pass_receive() -> NoReturn:
    await asyncio.sleep(10.0)
    return {"type": "http.request"}


async def async_generate_task(request, app):
    if request["request_type"] == "generate":
        if is_chat_completion_task(request):
            return await async_chat_completion_task(request, app)
        else:
            return await async_completion_task(request, app)
    else:
        raise ValueError(f"Invalid request type: {request['request_type']}")


def is_chat_completion_task(request):
    if isinstance(request["prompt"], str):
        return False

    return True


async def async_chat_completion_task(request, app):
    completion_request = ChatCompletionRequest(
        model=request["model"],
        messages=convert_prompt_to_openai_format(request["prompt"]),
        max_tokens=request["max_tokens"],
        temperature=0.0,
    )

    raw_request = Request(
        scope={"app": app, "type": "http", "headers": {}, "path": "/v1/completions"},
        receive=pass_receive,
    )

    response = await create_chat_completion(completion_request, raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {
        "request_id": request["request_id"],
    }

    if "choices" in response_data:
        response["response"] = response_data["choices"][0]["message"]["content"]
    elif response_data["object"] == "error":
        response["error"] = response_data["message"]

    if "usage" in response_data:
        response["token_count"] = response_data["usage"]["total_tokens"]
        response["flop_count"] = (
            compute_flop_count(await app.state.engine_client.get_model_config())
            * response_data["usage"]["total_tokens"]
        )

    await app.state.engine_client.check_health()

    return response


def convert_prompt_to_openai_format(
    prompt: PromptType,
) -> list[ChatCompletionMessageParam]:
    """Convert a prompt to OpenAI format."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": [{"role": "user", "content": prompt}]}]
    elif isinstance(prompt, dict):
        list_of_content = []
        for key, value in prompt.items():
            if key == "text":
                list_of_content.append({"type": "text", "text": value})
            elif key == "images":
                list_of_content.extend(
                    [{"type": "image_url", "image_url": {"url": image}} for image in value]
                )
            else:
                raise ValueError(f"Invalid prompt sub-field: {key}. Must be 'text' or 'image'.")
        return [{"role": "user", "content": list_of_content}]
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")


def compute_flop_count(model_config):

    # The intermediate size is the size of the feedforward layer
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    head_size = model_config.get_head_size()

    num_layers = getattr(model_config.hf_text_config, "num_hidden_layers", 12)
    num_attention_heads = getattr(model_config.hf_text_config, "num_attention_heads", 12)
    num_kv_heads = model_config.get_total_num_kv_heads()

    intermediate_size = getattr(model_config.hf_text_config, "intermediate_size", 4 * hidden_size)

    q_proj_flops = hidden_size * (num_attention_heads * head_size)
    kv_proj_flops = hidden_size * (num_kv_heads * head_size * 2)  # K and V

    # Attention computation: Q @ K^T and Attention @ V
    # Q @ K^T: [batch, heads, seq_len, head_size] @ [batch, heads, head_size, kv_cache_len]
    qk_flops = num_attention_heads * head_size

    # Attention @ V: [batch, heads, seq_len, kv_cache_len] @ [batch, heads, kv_cache_len, head_size]
    av_flops = num_attention_heads * head_size

    # Output projection: [batch, seq_len, hidden] @ [hidden, hidden]
    o_proj_flops = hidden_size * hidden_size

    attention_flops_per_layer = q_proj_flops + kv_proj_flops + qk_flops + av_flops + o_proj_flops
    total_attention_flops = attention_flops_per_layer * num_layers

    fc1_flops = hidden_size * intermediate_size
    fc2_flops = intermediate_size * hidden_size
    mlp_flops_per_layer = fc1_flops + fc2_flops

    total_mlp_flops = mlp_flops_per_layer * num_layers

    output_projection_flops = hidden_size * vocab_size

    embedding_flops = hidden_size * vocab_size

    total_flops = (
        total_attention_flops + total_mlp_flops + embedding_flops + output_projection_flops
    )

    return total_flops


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

    response = await create_completion(completion_request, raw_request=raw_request)

    response_data = json.loads(response.body.decode("utf-8"))

    logger.info("Got response: %s", response_data)

    response = {"request_id": request["request_id"]}

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

    await app.state.engine_client.check_health()

    return response

def kill_vllm_container():
    # Kill instances of pt_thread_main process
    os.system("pgrep pt_main_thread | xargs kill -9")
    os.system("pgrep python | xargs kill -9")
    sys.exit(1)
