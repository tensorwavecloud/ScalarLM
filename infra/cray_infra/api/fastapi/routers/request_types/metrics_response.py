from pydantic import BaseModel


class MetricsResponse(BaseModel):
    queue_depth: int
    total_completed_requests: int
    total_completed_tokens: int
    total_completed_response_time: float
    tokens_per_second: float
    requests_per_second: float
    flops_per_second: float

