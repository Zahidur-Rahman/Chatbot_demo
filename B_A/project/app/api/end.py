from fastapi import APIRouter, Depends
from app.services.llm import llm  # Import the initialized LLM

router = APIRouter()

@router.post("/generate/")
async def generate(prompt: str):
    try:
        response = llm.invoke(prompt)
        return {"response": response.content} # Access the text content
    except Exception as e:
        return {"error": f"LLM generation failed: {str(e)}"}