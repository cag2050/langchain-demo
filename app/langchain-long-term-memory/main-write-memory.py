from dotenv import load_dotenv
from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

load_dotenv()


@dataclass
class Context:
    user_id: str


class UserInfo(TypedDict):
    name: str


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    assert runtime.store is not None
    store = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, dict(user_info))
    return "Successfully saved user info."


store = InMemoryStore()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
)

agent = create_agent(
    model=llm,
    tools=[save_user_info],
    store=store,
    context_schema=Context,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith."}]},
    context=Context(user_id="user_123")
)
for message in response["messages"]:
    message.pretty_print()

item = store.get(("users",), "user_123")
print(f"item: {item}")
