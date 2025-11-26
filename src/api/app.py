from fastapi import FastAPI
from .routers.chat import chat_router

app = FastAPI(title="Q&A Agent API")

# include routers
app.include_router(chat_router)
