from langchain_core.tools import tool

from src.vector_store import search


@tool
def retrieve_from_knowledge_base(question: str) -> str:
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


