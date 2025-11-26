from fastapi import FastAPI
from .routers.chat import chat_router
from .routers.knowledge import knowledge_router
from src.db import init_db
from src.vector_store import load_existing_files

app = FastAPI(title="Q&A Agent API")

# include routers
app.include_router(chat_router)
app.include_router(knowledge_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and load knowledge files on startup."""
    init_db()
    load_existing_files()
