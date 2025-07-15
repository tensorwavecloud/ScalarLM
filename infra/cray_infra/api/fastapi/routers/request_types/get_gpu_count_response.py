from pydantic import BaseModel

class GetGPUCountResponse(BaseModel):
    gpu_count: int

