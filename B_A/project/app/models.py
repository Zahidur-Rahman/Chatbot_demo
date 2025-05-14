from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str
    session_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str | None = None