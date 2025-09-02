from pydantic import BaseModel
from cray_infra.api.fastapi.routers.request_types.get_adaptors_response import (
    GetAdaptorsResponse,
)

from typing import Optional, Union

PromptType = Union[str, dict[str, Union[str, list[str]]]]

class GetWorkResponse(BaseModel):
    prompt: PromptType
    request_id: int
    request_type: str
    model: Optional[str] = None
    max_tokens: Optional[int] = None


class GetWorkResponses(BaseModel):
    requests: list[GetWorkResponse]
    new_adaptors: GetAdaptorsResponse
