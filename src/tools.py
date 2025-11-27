from langchain_core.tools import tool

from src.vector_store import search


@tool(description=(
    "Search the local knowledge base for information relevant to a question. "
    "Use this tool when you need to find specific information that might be stored "
    "in the knowledge base, such as facts, details, or context about topics that "
    "have been uploaded to the system. The tool performs semantic search to find "
    "the most relevant document chunks. If the question is about general knowledge "
    "that you already know well, or if it's a simple question that doesn't require "
    "specific stored information, you may not need to use this tool. "
    "Input: the user's question or a search query related to what information is needed."
))
def retrieve_from_knowledge_base(question: str) -> str:
    """Retrieve relevant documents from the knowledge base using semantic search."""
    results = search(query=question, k=5)
    
    if not results:
        return "No relevant information found in the knowledge base. The knowledge base may be empty or the question doesn't match any stored content."
    
    formatted_results = []
    for chunk in results:
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        
        source_display = source[:8] + "..." if len(source) > 8 else source
        formatted_results.append(f"From document {source_display}:\n{content}")
    
    return "\n\n---\n\n".join(formatted_results)


