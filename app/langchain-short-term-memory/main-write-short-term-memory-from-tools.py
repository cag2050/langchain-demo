from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv()


class CustomAgentState(AgentState):
    user_name: str


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_info(runtime: ToolRuntime[CustomContext, CustomAgentState]) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id,
            )
        ]
    })


@tool
def greet(runtime: ToolRuntime[CustomContext, CustomAgentState]) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        })
    return f"Hello, {user_name}!"


llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model=llm,
    tools=[update_user_info, greet],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState,
    context_schema=CustomContext,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
    config=config,
)
for msg in response["messages"]:
    msg.pretty_print()
