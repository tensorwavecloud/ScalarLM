from cray_infra.api.fastapi.routers.openai_router import (
    openai_router,
)
from cray_infra.api.fastapi.routers.megatron_router import (
    megatron_router,
)
from cray_infra.api.fastapi.routers.health_router import (
    health_router,
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import logging
import os


logger = logging.getLogger(__name__)


app = FastAPI()

app.include_router(openai_router, prefix="/v1")
app.include_router(megatron_router, prefix="/v1")
app.include_router(health_router, prefix="/v1")

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

