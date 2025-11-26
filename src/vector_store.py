"""Vector store for RAG using ChromaDB and OpenAI embeddings."""

import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import API_KEY

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR),
    settings=Settings(anonymized_telemetry=False)
)

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)




def add_document(file_path: Path, content: str) -> tuple[bool, bool]:
    """Add a document to the vector store after chunking.
    
    Returns:
        (is_update, is_duplicate_content): 
        - is_update: True if file with same name already existed
        - is_duplicate_content: True if same content (hash) already exists with different name
    """
    # Generate file hash for content-based deduplication
    file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    
    # Check if file with same name already exists
    existing_by_name = collection.get(where={"file_name": file_path.name})
    is_update = len(existing_by_name["ids"]) > 0
    
    # Check if same content (hash) already exists with different filename
    existing_by_hash = collection.get(where={"file_hash": file_hash})
    is_duplicate_content = (
        len(existing_by_hash["ids"]) > 0 and 
        (not is_update or existing_by_hash["metadatas"][0][0].get("file_name") != file_path.name)
    )
    
    # Delete old chunks if updating by name
    if is_update:
        collection.delete(ids=existing_by_name["ids"])
    
    # Split document into chunks
    chunks = text_splitter.split_text(content)
    
    if not chunks:
        return (is_update, is_duplicate_content)
    
    # Generate embeddings for all chunks
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    # Prepare data for ChromaDB
    ids = [f"{file_path.stem}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "chunk_index": i,
            "file_hash": file_hash
        }
        for i in range(len(chunks))
    ]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=chunk_embeddings,
        documents=chunks,
        metadatas=metadatas
    )
    
    return (is_update, is_duplicate_content)


def search(query: str, k: int = 3) -> List[dict]:
    """Search the vector store for relevant chunks."""
    if collection.count() == 0:
        return []
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count())
    )
    
    # Format results
    retrieved_chunks = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
            retrieved_chunks.append({
                "content": doc,
                "source": metadata.get("file_name", "unknown"),
                "score": results["distances"][0][i] if results["distances"] and results["distances"][0] else None
            })
    
    return retrieved_chunks


def load_existing_files() -> None:
    """Load all existing .txt files from knowledge directory into vector store."""
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        return
    
    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8")
            add_document(file_path, content)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def delete_document(file_name: str) -> bool:
    """Delete a document and all its chunks from the vector store."""
    existing = collection.get(where={"file_name": file_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return True
    return False


def list_documents() -> List[str]:
    """List all unique document names in the vector store."""
    all_docs = collection.get()
    unique_files = set()
    if all_docs["metadatas"]:
        for metadata in all_docs["metadatas"]:
            if metadata and "file_name" in metadata:
                unique_files.add(metadata["file_name"])
    return sorted(list(unique_files))

