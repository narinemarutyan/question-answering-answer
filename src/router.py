from src.config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import BaseTool



result = agent.invoke({"input": "reverse hello"})
print(result)
