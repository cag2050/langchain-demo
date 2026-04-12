from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)

# Create a subagent
subagent = create_agent(model=llm)


# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    """research a topic and return findings"""
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# Main agent with subagent as a tool
main_agent = create_agent(model=llm, tools=[call_research_agent])

response = main_agent.invoke({"messages": [{"role": "user", "content": "3 + 2等于几？"}]})
print(response)
