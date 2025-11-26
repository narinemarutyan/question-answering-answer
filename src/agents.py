from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config import API_KEY
from src.tools import retrieve_from_knowledge_base


llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=API_KEY,
    temperature=0,
)

# Two agents:
# - one plain LLM without tools
# - one that can optionally call the local KB retrieval tool
plain_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful LLM agent, answering questions.",
)

kb_agent = create_agent(
    model=llm,
    tools=[retrieve_from_knowledge_base],
    system_prompt=(
        "You are a helpful LLM agent, answering questions. "
        "You have access to a local text knowledge base via the "
        "`retrieve_from_knowledge_base` tool; call it whenever it would improve "
        "your answer."
    ),
)


