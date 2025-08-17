from cray_infra.training.squeue import squeue
from cray_infra.training.get_gpu_count import get_gpu_count
from cray_infra.training.get_node_count import get_node_count

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import logging
import subprocess

logger = logging.getLogger(__name__)

slurm_router = APIRouter(prefix="/slurm")


@slurm_router.get("/status")
async def slurm_status():
    """Get SLURM cluster status including queue information and resource counts."""
    try:
        squeue_info = await squeue()
        gpu_count = get_gpu_count()
        node_count = get_node_count()

        return {
            "queue": squeue_info.dict(),
            "resources": {
                "gpu_count": gpu_count,
                "node_count": node_count
            },
            "status": "active"
        }
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e)
            },
            status_code=500
        )


@slurm_router.get("/squeue")
async def get_squeue():
    """Get SLURM queue information."""
    return await squeue()


@slurm_router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a SLURM job by ID."""
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True,
            text=True,
            check=True
        )

        return {
            "status": "success",
            "message": f"Job {job_id} cancelled successfully",
            "job_id": job_id
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to cancel job {job_id}: {e.stderr}")
        return JSONResponse(
            content={
                "status": "error",
                "error": f"Failed to cancel job {job_id}: {e.stderr}",
                "job_id": job_id
            },
            status_code=400
        )
    except Exception as e:
        logger.exception(f"Error cancelling job {job_id}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "job_id": job_id
            },
            status_code=500
        )


@slurm_router.get("/endpoints")
async def list_routes():
    routes = [
        f"Path: {route.path}, Methods: {', '.join(route.methods)}"
        for route in slurm_router.routes
    ]
    return JSONResponse(content={"endpoints": routes}, media_type="application/json")
