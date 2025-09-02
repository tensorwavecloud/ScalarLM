# TODO: remove openai_router?
from cray_infra.api.fastapi.routers.openai_router import (
    openai_router,
)
from cray_infra.api.fastapi.routers.openai_v1_router import (
    openai_v1_router,
)
from cray_infra.api.fastapi.routers.megatron_router import (
    megatron_router,
)
from cray_infra.api.fastapi.routers.health_router import (
    health_router,
)
from cray_infra.api.fastapi.routers.generate_router import (
    generate_router,
)
from cray_infra.api.fastapi.routers.slurm_router import (
    slurm_router,
)
from cray_infra.api.fastapi.routers.work_router import (
    router as work_router,
)

from cray_infra.api.fastapi.tasks.add_megatron_tasks import (
    add_megatron_tasks,
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import logging
import os

logger = logging.getLogger(__name__)

# Initialize ScalarLM vLLM adapters for clean architecture
try:
    from cray_infra.adapters import init_adapters
    # Initialize the adapter system on startup
    init_adapters()
    logger.info("✅ ScalarLM adapters initialized successfully")
except ImportError as e:
    logger.warning(f"⚠️  ScalarLM vLLM adapters not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to initialize ScalarLM vLLM adapters: {e}")


app = FastAPI(lifespan=add_megatron_tasks)

app.include_router(openai_router, prefix="/v1")
app.include_router(openai_v1_router, prefix="/v1")
app.include_router(megatron_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")
app.include_router(generate_router, prefix="/v1")
app.include_router(work_router)  # Already includes /v1 prefix
app.include_router(slurm_router)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
