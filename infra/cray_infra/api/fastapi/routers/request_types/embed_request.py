from pydantic import BaseModel

from typing import Optional


class EmbedRequest(BaseModel):
    model: Optional[str] = None
    prompts: list[str]

