"""Vector store for RAG using ChromaDB and OpenAI embeddings."""

import hashlib
import os
from pathlib import Path
from typing import List

# Disable ChromaDB telemetry before importing
os.environ["ANONYMIZED_TELEMETRY"] = "False"

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


def _get_content_hash(content: str) -> str:
    """Generate a hash for content to use as document identifier."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def add_document(content: str, file_name: str = None) -> str:
    """Add a document to the vector store after chunking.
    
    Returns the content hash that identifies this document.
    """
    # Generate content hash as the document identifier
    doc_hash = _get_content_hash(content)
    
    # Check if document with this hash already exists
    existing = collection.get(where={"doc_hash": doc_hash})
    if existing["ids"]:
        # Delete old chunks for this document
        collection.delete(ids=existing["ids"])
    
    # Split document into chunks
    chunks = text_splitter.split_text(content)
    
    if not chunks:
        return doc_hash
    
    # Generate embeddings for all chunks
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    # Prepare data for ChromaDB
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_hash": doc_hash,
            "file_name": file_name or f"doc_{doc_hash[:8]}.txt",
            "chunk_index": i,
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
    
    return doc_hash


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
                "source": metadata.get("doc_hash", "unknown"),
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
            add_document(content, file_name=file_path.name)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def delete_document(doc_hash: str) -> bool:
    """Delete a document and all its chunks from the vector store by hash."""
    existing = collection.get(where={"doc_hash": doc_hash})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return True
    return False


def document_exists(doc_hash: str) -> bool:
    """Check if a document with the given hash exists."""
    existing = collection.get(where={"doc_hash": doc_hash})
    return len(existing["ids"]) > 0


def list_documents() -> List[str]:
    """List all unique document hashes in the vector store."""
    all_docs = collection.get()
    unique_hashes = set()
    if all_docs["metadatas"]:
        for metadata in all_docs["metadatas"]:
            if metadata and "doc_hash" in metadata:
                unique_hashes.add(metadata["doc_hash"])
    return sorted(list(unique_hashes))

