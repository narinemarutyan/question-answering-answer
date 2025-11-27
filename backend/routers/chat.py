from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.agents import kb_agent
from backend.db import add_message, delete_session, get_messages, list_sessions, session_exists

chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    chat_id: int


class ChatQuestionRequest(BaseModel):
    chat_id: int
    question: str


@chat_router.get("/list")
def list_chat_sessions():
    """List all chat sessions."""
    return {"sessions": list_sessions()}


@chat_router.post("/get_messages")
def get_chat_messages(payload: ChatRequest):
    """List messages for a specific chat session."""
    messages = get_messages(payload.chat_id)
    if not messages:
        raise HTTPException(
            status_code=404,
            detail="Chat not found or has no messages"
        )
    return {"chat_id": payload.chat_id, "messages": messages}


@chat_router.post("/answer")
def answer_chat_question(payload: ChatQuestionRequest):
    """Load history → use KB agent (LLM decides when to use KB tool) → get answer → store messages."""

    history = get_messages(payload.chat_id)

    agent = kb_agent

    messages = history + [
        {"role": "user", "content": payload.question}
    ]

    try:
        result = agent.invoke(
            {"messages": messages},
            context={"user_role": "expert"},
        )

        reply = result["messages"][-1].content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save both messages to DB
    add_message(payload.chat_id, "user", payload.question)
    add_message(payload.chat_id, "assistant", reply)

    return {
        "chat_id": payload.chat_id,
        "question": payload.question,
        "answer": reply,
    }


@chat_router.delete("/delete", status_code=204)
def delete_chat(chat_id: int):
    """Delete a chat session and all its messages. Works even if session doesn't exist (orphaned messages)."""
    # Check if there are any messages for this chat_id
    messages = get_messages(chat_id)
    if not messages and not session_exists(chat_id):
        raise HTTPException(
            status_code=404,
            detail=f"Chat with ID {chat_id} not found. Use GET /chat/list to see available sessions."
        )

    delete_session(chat_id)

    return None

