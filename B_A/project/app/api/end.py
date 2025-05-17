from fastapi import APIRouter, Depends, HTTPException
from app.services.rag import RAGService
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 6# Optional parameter for number of chunks to retrieve

@router.post("/query/")
async def query(request: QueryRequest):
    try:
        result = rag_service.query(request.question, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))