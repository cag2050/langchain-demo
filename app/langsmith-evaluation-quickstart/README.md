> 参考：https://docs.langchain.com/langsmith/evaluation-quickstart

步骤：
1. 安装openevals：`pip install -U openevals -i https://pypi.tuna.tsinghua.edu.cn/simple`
2. 运行：dataset.py，会将数据上传到自己账号：https://smith.langchain.com/ 的左侧导航：Datasets & Experiments 里。
3. 运行：eval.py（target函数里，已经将调用：OpenAI，改为调用：DeepSeek），运行成功后，会有一个评估结果的LangSmith链接。左侧导航：Datasets & Experiments 里表格，有一列：Experiments，显示的数值是实验次数。