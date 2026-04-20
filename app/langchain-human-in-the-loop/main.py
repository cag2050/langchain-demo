from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langgraph.types import Command

load_dotenv()

llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)


@tool
def write_file_tool():
    """write_file_tool"""
    return "write_file_tool"


@tool
def execute_sql_tool():
    """execute_sql_tool"""
    return "execute_sql_tool"


@tool
def read_data_tool():
    """read_data_tool"""
    return "read_data_tool"


agent = create_agent(
    model=llm,
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file_tool": True,  # 暂停并等待：允许 approve、edit、reject 三种操作。
                "execute_sql_tool": {
                    "allowed_decisions": ["approve", "reject"]  # 暂停并等待：允许数组里面的操作
                },
                "read_data_tool": False  # 直接执行：不中断，直接执行工具
            },
            description_prefix="Tool execution pending approval"
        ),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "some_id"}}

# 手动注入一个伪造的 AI 响应
# 我们直接告诉 Agent：“模型刚刚决定调用 execute_sql_tool 工具”
# AIMessage 和 ToolMessage 是一对“请求与响应”的搭档。
fake_ai_messages = [
    AIMessage(
        content="",  # 内容可以为空
        tool_calls=[
            {
                "name": "execute_sql_tool",  # 必须匹配你配置中需要中断的工具名
                "args": {"query": "DELETE FROM users WHERE id = 1"},
                "id": "call_mock_123"  # 必须提供唯一的 ID
            }
        ]
    ),
    ToolMessage(content="make a error", tool_call_id="call_mock_123")
]

response1 = agent.invoke(
    {"messages": fake_ai_messages},
    config=config,
    version="v2",
)

print(f"response1: {response1}\n")
print(f"response1.interrupts: {response1.interrupts}\n")

# Resume with approval decision
response2 = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "reject"
    ),
    config=config,  # Same thread ID to resume the paused conversation
    version="v2",
)
print(f"response2: {response2}\n")
