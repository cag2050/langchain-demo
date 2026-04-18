from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client
# convert_to_messages：将外部数据（如字符串、字典列表、OpenAI 格式数据）转换为 LangChain 标准的消息对象（ from langchain_core.messages.base import BaseMessage 子类）。
# convert_to_openai_messages：将 LangChain 标准的消息对象转换为 OpenAI API 兼容的字典格式。
from langchain_core.messages import convert_to_messages, convert_to_openai_messages

load_dotenv()

client = Client()
# 返回：from langchain_core.prompts import PromptTemplate
prompt = client.pull_prompt("prompt-quickstart")
print(f"prompt: {prompt}\n")

# 返回：from langchain_core.prompt_values import PromptValue ，有方法：def to_messages(self) -> list[BaseMessage]:
formatted_prompt = prompt.invoke({"question": "What is the color of the sky?"})
print(f"formatted_prompt: {formatted_prompt}\n")

# 返回：dict | list[dict]，OpenAI message dicts。
openai_messages = convert_to_openai_messages(formatted_prompt)
print(f"openai_messages: {openai_messages}\n")

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek"
)

# 返回：from langchain_core.messages.ai import AIMessage
response = model.invoke(openai_messages)
# response = model.invoke(formatted_prompt) # 或者直接传：from langchain_core.prompt_values import PromptValue
print(f"response: {response}\n")
