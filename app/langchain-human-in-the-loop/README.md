> https://docs.langchain.com/oss/python/langchain/human-in-the-loop

### 在流式场景下，人类决策传回程序的步骤是：
1. 前端/控制台：通过 stream 监听到 __interrupt__ 事件。
2. 用户：做出决策（批准/修改）。
3. 程序：发起一次新的 agent.invoke(Command(resume=决策), config=config) 调用。 
4. 关键点： 必须保证两次调用使用了同一个 thread_id（在 config 中），否则系统找不到之前的暂停点。