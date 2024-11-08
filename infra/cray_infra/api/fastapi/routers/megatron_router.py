import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

megatron_router = APIRouter(prefix="/megatron")
