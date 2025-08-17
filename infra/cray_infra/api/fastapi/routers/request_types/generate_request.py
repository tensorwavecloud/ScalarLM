from pydantic import BaseModel

from typing import Optional, Union


class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompts: list[Union[str, dict[str, Union[str, list[str]]]]]
    max_tokens: Optional[int] = 16