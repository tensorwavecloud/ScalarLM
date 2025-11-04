from pydantic import BaseModel

class DownloadRequest(BaseModel):
    request_id: str

