
import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

from cray_infra.one_server.start_cray_server import start_cray_server

from uvicorn.supervisors import ChangeReload
import uvicorn

import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)


def main():
    try:
        return run_server_with_autoreload()

    except Exception as e:
        print(e)
        sys.exit(0)


def run_server_with_autoreload():

    server_config = uvicorn.Config(
        "cray_infra.one_server.main:run_all_servers",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload_dirs=["/app/cray/infra/cray_infra"],
        reload_excludes=["**/jobs/**/*.yaml"],
        reload=True,
        reload_includes=["**/*.py", "**/*.yaml"],
    )

    sock = server_config.bind_socket()

    supervisor = ChangeReload(server_config, target=run_all_servers, sockets=["8000"])

    supervisor.run()


def run_all_servers(sockets):
    asyncio.run(run_all_servers_async())


async def run_all_servers_async():
    server_status = await start_cray_server(server_list=["all"])

    done, pending = await asyncio.wait(
        server_status.tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )

    logger.info("Cray sever is shutting down")
    for pending_task in pending:
        pending_task.cancel("Another service died, server is shutting down")


if __name__ == "__main__":
    main()
