from langchain_core.tools import tool

from src.vector_store import search


@tool
def retrieve_from_knowledge_base(question: str) -> str:
    """Look up information in the local knowledge base using semantic search (RAG).
    
    This tool uses vector embeddings to find the most relevant chunks from the 
    knowledge base that match the question semantically. Returns the top relevant 
    excerpts with their source file names.
    """
    # Use semantic search to find relevant chunks
    results = search(query=question, k=5)
    
    if not results:
        return "No relevant information found in the knowledge base. The knowledge base may be empty or the question doesn't match any stored content."
    
    # Format results with source information
    formatted_results = []
    for chunk in results:
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        formatted_results.append(f"From {source}:\n{content}")
    
    return "\n\n---\n\n".join(formatted_results)


