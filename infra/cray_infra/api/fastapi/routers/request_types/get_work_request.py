from pydantic import BaseModel

from typing import Optional

class GetWorkRequest(BaseModel):
    batch_size: int
    loaded_adaptor_count: int

