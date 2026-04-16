from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)


class CustomAgentState(AgentState):
    user_id: int
    user_name: str


@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """Look up the user information."""
    user_id = runtime.state["user_id"]
    user_name = runtime.state["user_name"]
    return f"User {user_id}: {user_name}"


agent = create_agent(
    model=llm,
    tools=[
        get_user_info,
    ],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

response_1 = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hi! My name is Bob."}],
        "user_id": 1,
        "user_name": "Bob"
    },
    config=config,
)
for msg in response_1["messages"]:
    msg.pretty_print()

print(">>>>>>>>>>>>>>>>>>>>>>>>>")

response_2 = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is my name?"}],
    },
    config=config,
)
for msg in response_2["messages"]:
    msg.pretty_print()
