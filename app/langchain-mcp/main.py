import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",  # Local subprocess communication
                "command": "python",
                # Absolute path to your math_server.py file
                "args": ["/path/to/math_server.py"],  # todo 修改成本地绝对路径
            },
            "weather": {
                "transport": "streamable_http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                # 先启动：weather.py
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    tools = await client.get_tools()
    llm = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek"
    )
    agent = create_agent(
        llm,
        tools
    )

    math_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what's (3 + 5) x 12?"
                }
            ]
        }
    )
    weather_response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "what is the weather in nyc?"
                }
            ]
        }
    )
    print(math_response)
    print(weather_response)


if __name__ == "__main__":
    asyncio.run(main())
