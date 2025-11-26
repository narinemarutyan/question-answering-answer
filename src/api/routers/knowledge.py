"""Knowledge base management endpoints for RAG."""

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.vector_store import add_document, delete_document, list_documents

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

knowledge_router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class DocumentListResponse(BaseModel):
    documents: List[str]


@knowledge_router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a .txt file to the knowledge base.
    
    The file will be chunked, embedded, and added to the vector store for RAG retrieval.
    """
    # Validate file extension
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    # Read file content
    try:
        content = await file.read()
        text_content = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading file: {str(e)}"
        )
    
    # Save file to knowledge directory
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = KNOWLEDGE_DIR / file.filename
    
    try:
        file_path.write_text(text_content, encoding='utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Add to vector store
    try:
        is_update, is_duplicate_content = add_document(file_path, text_content)
    except Exception as e:
        # If vector store fails, still keep the file
        raise HTTPException(
            status_code=500,
            detail=f"Error adding to vector store: {str(e)}"
        )
    
    # Provide informative message based on what happened
    if is_update:
        message = f"File '{file.filename}' updated successfully (replaced existing file with same name)"
    elif is_duplicate_content:
        message = f"File '{file.filename}' uploaded successfully (note: same content already exists with different filename)"
    else:
        message = f"File '{file.filename}' uploaded and processed successfully"
    
    return {
        "message": message,
        "filename": file.filename,
        "is_update": is_update,
        "is_duplicate_content": is_duplicate_content
    }


@knowledge_router.get("/list", response_model=DocumentListResponse)
def list_knowledge_documents():
    """List all documents in the knowledge base."""
    documents = list_documents()
    return DocumentListResponse(documents=documents)


@knowledge_router.delete("/delete/{filename}")
def delete_knowledge_document(filename: str):
    """Delete a document from the knowledge base.
    
    This removes the file and all its chunks from the vector store.
    """
    # Validate filename (security: prevent path traversal)
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = KNOWLEDGE_DIR / filename
    
    # Delete from vector store
    deleted_from_store = delete_document(filename)
    
    # Delete file if it exists
    file_deleted = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_deleted = True
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting file: {str(e)}"
            )
    
    if not deleted_from_store and not file_deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    return {
        "message": f"Document '{filename}' deleted successfully"
    }

