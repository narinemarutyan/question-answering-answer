# src/api/routers/knowledge.py

from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from src.vector_store import add_document, delete_document, list_documents

knowledge_router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ---------------------------
# Request/Response Models
# ---------------------------

class DocumentListResponse(BaseModel):
    documents: list[str]


class DeleteDocumentRequest(BaseModel):
    file_name: str


class AddDocumentRequest(BaseModel):
    file_name: str
    content: str


class DeleteDocumentResponse(BaseModel):
    success: bool
    message: str


class DocumentResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    is_duplicate: bool


# ---------------------------
# Endpoints
# ---------------------------

@knowledge_router.get("/list", response_model=DocumentListResponse)
def list_knowledge_documents():
    """List all documents in the knowledge base."""
    documents = list_documents()
    return {"documents": documents}


@knowledge_router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a text file to the knowledge base.
    
    The file will be processed, chunked, and added to the vector store.
    If a file with the same name already exists, it will be replaced.
    """
    # Only accept .txt files
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Create file path in knowledge directory
        from src.vector_store import KNOWLEDGE_DIR
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = KNOWLEDGE_DIR / file.filename
        
        # Check if document already exists
        is_duplicate = file_path.exists() or file.filename in list_documents()
        
        # Save file to disk
        file_path.write_text(content_str, encoding='utf-8')
        
        # Add to vector store (will replace if exists)
        add_document(file_path, content_str)
        
        message = (
            f"Document '{file.filename}' updated and re-indexed successfully"
            if is_duplicate
            else f"Document '{file.filename}' uploaded and indexed successfully"
        )
        
        return {
            "success": True,
            "message": message,
            "file_name": file.filename,
            "is_duplicate": is_duplicate
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@knowledge_router.post("/add", response_model=DocumentResponse)
def add_document_text(payload: AddDocumentRequest):
    """Add a document to the knowledge base by providing text content directly.
    
    Request body:
    {
        "file_name": "example.txt",
        "content": "The text content of the document..."
    }
    """
    file_name = payload.file_name
    content = payload.content
    
    if not file_name.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="File name must end with .txt"
        )
    
    try:
        from src.vector_store import KNOWLEDGE_DIR
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = KNOWLEDGE_DIR / file_name
        
        # Check if document already exists
        is_duplicate = file_path.exists() or file_name in list_documents()
        
        # Save file to disk
        file_path.write_text(content, encoding='utf-8')
        
        # Add to vector store (will replace if exists)
        add_document(file_path, content)
        
        message = (
            f"Document '{file_name}' updated and re-indexed successfully"
            if is_duplicate
            else f"Document '{file_name}' added and indexed successfully"
        )
        
        return {
            "success": True,
            "message": message,
            "file_name": file_name,
            "is_duplicate": is_duplicate
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error adding document: {str(e)}"
        )


@knowledge_router.delete("/delete", response_model=DeleteDocumentResponse)
def delete_knowledge_document(payload: DeleteDocumentRequest):
    """Delete a document from the knowledge base."""
    file_name = payload.file_name
    
    # Delete from vector store
    deleted = delete_document(file_name)
    
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{file_name}' not found in knowledge base"
        )
    
    # Also delete from disk
    try:
        from src.vector_store import KNOWLEDGE_DIR
        file_path = KNOWLEDGE_DIR / file_name
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        # If file deletion fails, that's okay - it's already removed from vector store
        pass
    
    return {
        "success": True,
        "message": f"Document '{file_name}' deleted successfully"
    }

