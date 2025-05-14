from fastapi import Depends
from app.services.rag import RAGService

def get_rag_service() -> RAGService:
    return RAGService()

# Example usage in endpoints:
# async def endpoint(rag: RAGService = Depends(get_rag_service))