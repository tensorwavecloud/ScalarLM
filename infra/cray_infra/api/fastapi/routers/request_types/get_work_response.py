from pydantic import BaseModel

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
