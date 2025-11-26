from fastapi import FastAPI

from .routers.chat import chat_router
from .routers.knowledge import knowledge_router

app = FastAPI(title="Q&A Agent API")


@app.on_event("startup")
async def startup_event():
    """Initialize vector store and load existing knowledge files on startup."""
    from src.vector_store import load_existing_files
    load_existing_files()


# Include routers
app.include_router(chat_router)
app.include_router(knowledge_router)
