from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

embeddings = DashScopeEmbeddings(model="text-embedding-v3")

vector_store = InMemoryVectorStore(embedding=embeddings)

docs = [
    Document(page_content="LangChain 是一个构建大模型应用的框架", metadata={"source": "tech_blog"}),
    Document(page_content="向量数据库用于存储嵌入向量", metadata={"source": "tech_blog"}),
]
vector_store.add_documents(documents=docs, ids=["id1", "id2"])

similar_docs = vector_store.similarity_search("langchain")
print(f"similar_docs: {similar_docs}")

# 定义过滤逻辑：如果文档的 source 元数据等于 'tech_blog'，则返回 True
my_filter = lambda doc: doc.metadata.get("source") == "tech_blog"
# 注意参数filter类型：filter: Callable[[Document], bool] | None = None,
similar_docs_with_filter = vector_store.similarity_search(
    "langchain",
    k=3,
    filter=my_filter,
)
print(f"similar_docs_with_filter: {similar_docs_with_filter}")
