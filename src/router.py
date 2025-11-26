from typing import Literal


RouteDecision = Literal["kb", "llm"]


def needs_knowledge_base(question: str) -> bool:
    """Very simple router heuristic.

    Right now we just key off words that are likely to require the local animal KB.
    You can expand this logic later (regexes, ML model, extra metadata, etc.).
    """
    q = question.lower()
    kb_keywords = [
        "koala",
        "lion",
        "rabbit",
        "animal",
        "habitat",
        "species",
    ]
    return any(keyword in q for keyword in kb_keywords)


def route_question(question: str) -> RouteDecision:
    """Return 'kb' if the question should consult the local knowledge base, else 'llm'."""
    return "kb" if needs_knowledge_base(question) else "llm"


