from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.config import API_KEY

llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=API_KEY,
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a helpful LLM agent, answering questions."
)

chat_history = []

while True:
    user_input = input("You: ".strip())
    messages = chat_history + [{"role": "user", "content": user_input}]
    result = agent.invoke(
        {"messages": messages},
        context={"user_role": "expert"}
    )
    try:
        reply = result["messages"][-1]["content"]
    except Exception as e:
        reply = str(e)

    print(reply)
    chat_history.append({"role": "user", "content": reply})
