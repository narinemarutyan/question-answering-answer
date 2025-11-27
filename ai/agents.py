from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from ai.config import API_KEY
from ai.tools import retrieve_from_knowledge_base

llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=API_KEY,
    temperature=0,
)

kb_agent = create_agent(
    model=llm,
    tools=[retrieve_from_knowledge_base],
    system_prompt=(
        "You are a helpful LLM agent that answers questions accurately and thoroughly. "
        "You have access to a local knowledge base through the `retrieve_from_knowledge_base` tool. "
        "Read the tool's description carefully to understand when it would be useful. "
        "Use the tool when the question requires specific information that might be stored "
        "in the knowledge base. If the question is about general knowledge you already know, "
        "or if you're confident in your answer without needing stored documents, you can "
        "answer directly. Always prioritize accuracy - if you're unsure or the question "
        "seems to require specific stored information, use the tool to search the knowledge base."
    ),
)

