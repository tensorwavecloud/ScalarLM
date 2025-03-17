from masint.util.make_api_url import make_api_url

import masint

import aiohttp
import asyncio
import json
import jsonlines
import io

import logging

logger = logging.getLogger(__name__)


def view_logs(model_name: str, tail: bool, lines: int, follow: bool):
    logger.debug(f"Viewing logs for model {model_name}")

    asyncio.run(
        view_logs_async(model_name=model_name, tail=tail, lines=lines, follow=follow)
    )


async def view_logs_async(model_name: str, tail: bool, lines: int, follow: bool):
    already_printed_lines = set()

    starting_line_number = 0

    while True:
        logs = await sample_log_stream(
            model_name=model_name, starting_line_number=starting_line_number
        )

        if tail:
            logs = logs[-lines:]

        for log in logs:
            if log["line_number"] in already_printed_lines:
                continue

            print(format_log(log))
            already_printed_lines.add(log["line_number"])

            if log["line_number"] > starting_line_number:
                starting_line_number = log["line_number"]

        if not follow:
            break

        await asyncio.sleep(1)


async def sample_log_stream(model_name: str, starting_line_number: int):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            make_api_url(f"v1/megatron/train/logs/{model_name}"),
            params={"starting_line_number": starting_line_number},
        ) as resp:
            if resp.content_type != "text/event-stream":
                # The response is json
                response = await resp.json()

                raise Exception(f"Error {resp.status} getting log stream: {response}")

            object_buffer = await read_log_stream(resp.content)

    return object_buffer


async def read_log_stream(log_stream):
    object_buffer = []

    async for chunk in log_stream.iter_any():
        text = chunk.decode("utf-8")

        try:
            reader = jsonlines.Reader(io.StringIO(text))

            for obj in reader:
                object_buffer.append(obj)
        except json.JSONDecodeError:
            logger.debug(f"Failed to decode json: {text}")
            continue

    return object_buffer


def format_log(log):
    return f"{log['line_number']} {log['line'].rstrip()}"
