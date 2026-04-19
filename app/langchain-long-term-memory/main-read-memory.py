from dotenv import load_dotenv
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain.tools import ToolRuntime, tool

load_dotenv()


@dataclass
class Context:
    user_id: str


store = InMemoryStore()
store.put(
    ("users",),
    "user_123",
    {
        "name": "John Smith",
        "language": "English"
    }
)


@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    user_info = runtime.store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"


llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)

agent = create_agent(
    model=llm,
    tools=[get_user_info],
    store=store,
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user info"}]},
    context=Context(user_id="user_123"),
)

for message in response["messages"]:
    message.pretty_print()
