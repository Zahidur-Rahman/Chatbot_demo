from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import AsyncIterator
from app.api.end import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup code
    print("Initializing services...")
    yield
    # Shutdown code
    print("Cleaning up resources...")

app = FastAPI(
    title="Mistral API Example",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router)