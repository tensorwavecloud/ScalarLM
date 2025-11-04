from masint.util.make_api_url import make_api_url

import aiohttp
import json
import tempfile
import logging

logger = logging.getLogger(__name__)


async def poll_for_downloads(result, api_url=None):
    api_url = make_api_url("v1/generate/download", api_url=api_url)

    request_id = result["request_id"]

    async with aiohttp.ClientSession() as session:
        final_result = None

        while not is_download_finished(final_result):
            async with session.post(api_url, json={"request_id": request_id}) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
                    logger.debug(f"Created temporary download file at {f.name}")

                    async for chunk in response.content.iter_chunked(64 * 1024):
                        if chunk:
                            f.write(chunk.decode("utf-8"))
                            downloaded += len(chunk)
                            logger.debug(
                                f"Downloaded {downloaded} of {total_size} bytes"
                                " ({(downloaded/total_size)*100:.2f}%)"
                            )

                    f.flush()
                    f.seek(0)

                    final_result = json.load(f)

    return final_result


def is_download_finished(result):
    if result is None:
        return False

    logger.debug(f"Download result status: {result['status']}")

    if result["status"] != "completed":
        return False

    for response in result["results"]:
        if response["error"] is not None:
            raise Exception(response["error"])

    return True
