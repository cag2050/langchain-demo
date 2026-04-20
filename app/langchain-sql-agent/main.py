import pathlib

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# 下载SQLite数据
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    print(f"response: {response}\n")
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"db: {db}\n")
print(f"Available tables: {db.get_usable_table_names()}\n")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}\n')

# create_agent
model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
)

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "sql_db_query": True
            },
            description_prefix="Tool execution pending approval",
        )
    ],
    checkpointer=InMemorySaver(),
)

question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": 1}}

for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=config,
):
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass

# 不能直接在一个已经暂停的 stream 对象里“写入”数据。需要发起第二次调用。
# 当你调用 Command(resume=...) 时，LangGraph 的 Checkpointer 会根据 thread_id 找到上次暂停时的内存快照，把 human_decision 赋值给代码中 interrupt() 函数的返回值，然后从那一行代码继续往下执行。
for step in agent.stream(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
        stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass
