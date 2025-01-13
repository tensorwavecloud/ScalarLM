from cray_infra.api.fastapi.routers.request_types.squeue_response import SqueueResponse

import subprocess


async def squeue():
    try:
        squeue_output = subprocess.check_output(
            ["squeue", '--format=%.18i %.9P %.12j %.8u %.8T %.10M %.9l %.6D %R']
        )

        return SqueueResponse(
            squeue_output=squeue_output.decode("utf-8"),
        )

    except subprocess.CalledProcessError:
        return SqueueResponse(
            error_message="squeue command failed",
        )
