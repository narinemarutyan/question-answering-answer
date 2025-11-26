import os
from pathlib import Path

from langchain_core.tools import tool


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


@tool
def retrieve_from_knowledge_base(question: str) -> str:
    """Look up information in the local text knowledge base to help answer the question.

    The knowledge base consists of simple `.txt` files under the `knowledge` directory.
    This tool returns short, relevant excerpts (or all contents if the base is tiny).
    """
    if not KNOWLEDGE_DIR.exists():
        return "No local knowledge base found."

    question_lower = question.lower()
    snippets: list[str] = []

    for file in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        text_lower = text.lower()
        # Very simple relevance heuristic: check if any question word appears in the file.
        # If nothing matches, we still allow returning short contents for tiny KBs.
        if any(word in text_lower for word in question_lower.split()):
            snippets.append(f"From {file.name}:\n{text.strip()}")

    if not snippets:
        # Fallback: return all contents if the KB is small.
        all_texts: list[str] = []
        for file in KNOWLEDGE_DIR.glob("*.txt"):
            try:
                text = file.read_text(encoding="utf-8")
            except Exception:
                continue
            all_texts.append(f"From {file.name}:\n{text.strip()}")

        if not all_texts:
            return "Knowledge base is empty."

        return "\n\n".join(all_texts)

    return "\n\n".join(snippets)


