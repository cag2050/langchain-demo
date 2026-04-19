from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


llm = init_chat_model(
    # model="deepseek-reasoner",
    model="deepseek-chat",
    model_provider="deepseek",
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
)

for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in Beijing?"}]},
        # stream_mode="updates",  # StreamMode的种类：StreamMode = Literal["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"]
        stream_mode=["values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"],
        version="v2",
):
    # print(f"chunk: {chunk}\n")
    # print(f"chunk: {chunk['type']}")
    # print(f"data: {chunk['data']}\n")

    # if chunk["type"] == "updates":
    #     print("=====updates===")
    #     for step, data in chunk['data'].items():
    #         print(f"step: {step}")
    #         print(f"content: {data['messages'][-1].content}\n")
    #

    if chunk["type"] == "messages":
        print("=====messages===")
        print(f"chunk: {chunk}\n")
        token, metadata = chunk["data"]
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}\n")  # 聊天界面的“打字机效果”
    #
    # if chunk["type"] == "custom":
    #     print("=====custom===")
    #     print(chunk["data"])
