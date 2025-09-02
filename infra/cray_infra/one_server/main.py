import os

os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

from cray_infra.one_server.start_cray_server import start_cray_server
from cray_infra.util.get_config import get_config

from uvicorn.supervisors import ChangeReload
import uvicorn

import asyncio
import logging
import os
import sys
import traceback

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


def main():
    try:
        return run_server_with_autoreload()

    except Exception as e:
        print(e)
        sys.exit(0)


def run_server_with_autoreload():

    os.chdir("/app/cray/infra")

    server_config = uvicorn.Config(
        "cray_infra.one_server.main:run_all_servers",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload_dirs=["/app/cray/infra/cray_infra"],
        reload_excludes=["**/jobs/**"],
        reload=True,
        reload_includes=["**/*.py", "**/*.yaml"],
    )

    sock = server_config.bind_socket()

    supervisor = ChangeReload(server_config, target=run_all_servers, sockets=["8000"])

    supervisor.run()


def run_all_servers(sockets):
    asyncio.run(run_all_servers_async())


async def run_all_servers_async():
    config = get_config()

    server_status = await start_cray_server(server_list=[config["server_list"]])

    logger.info(f"Running with {len(server_status.tasks)} servers")

    if len(server_status.tasks) > 0:
        done, pending = await asyncio.wait(
            server_status.tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Check for exceptions in completed tasks
        for task in done:
            try:
                # This will raise the exception if the task failed
                result = task.result()
            except Exception as e:
                logger.error(f"Task {task.get_name()} failed with exception:")
                logger.error(traceback.format_exc())

        logger.info("Cray sever is shutting down")

        for pending_task in pending:
            try:
                pending_task.cancel("Another service died, server is shutting down")
            except Exception as e:
                logger.error(f"Error canceling task {pending_task.get_name()}:")
                logger.error(traceback.format_exc())
    else:
        try:
            while True:
                logger.info("Server is sleeping forever")
                await asyncio.sleep(600)
        except asyncio.CancelledError:
            logger.info("Sleep loop was cancelled")
            raise
        except Exception as e:
            logger.error("Error in sleep loop:")
            logger.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
