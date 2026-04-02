from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"It's always sunny in {city}"


llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "what is the weather in sf"
            }
        ],
    }
)

print(response["messages"][-1].content)
