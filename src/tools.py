from langchain_core.tools import tool

from src.vector_store import search


@tool
def retrieve_from_knowledge_base(question: str) -> str:
    """Look up information in the local knowledge base using semantic search.

    This tool uses RAG (Retrieval Augmented Generation) to find relevant information
    from uploaded documents. It performs semantic similarity search to retrieve the
    most relevant chunks of text related to the question.
    """
    results = search(question, k=3)
    
    if not results:
        return "No relevant information found in the knowledge base."
    
    # Format results with source attribution
    formatted_results = []
    for result in results:
        source = result.get("source", "unknown")
        content = result.get("content", "")
        formatted_results.append(f"From {source}:\n{content}")
    
    return "\n\n---\n\n".join(formatted_results)


