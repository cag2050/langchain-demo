>  LangGraph的定制SQL Agent，参考：https://docs.langchain.com/oss/python/langgraph/sql-agent

与 LangChain 的开箱即用SQL Agent（ https://docs.langchain.com/oss/python/langchain/sql-agent ） 不同，LangGraph的定制SQL Agent（ https://docs.langchain.com/oss/python/langgraph/sql-agent ） 允许开发者通过定义图（Graph）的节点（Nodes）和边（Edges）来实现对 Agent 行为的精细控制，特别是强制执行特定的工具调用流程和人工审核机制。

代码比较 | main.py | main-human-in-the-loop_agent_graph.py
--- |--------| ---
check_query节点 | ✅ | ⬜
run_query_tool_with_interrupt函数 | ⬜  | ✅