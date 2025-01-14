from pydantic import BaseModel


class ListModelsResponse(BaseModel):
    models: list[str]
