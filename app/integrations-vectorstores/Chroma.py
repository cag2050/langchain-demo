from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_chroma import Chroma

load_dotenv()

embeddings = DashScopeEmbeddings(model="text-embedding-v3")

vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

docs = [
    Document(page_content="LangChain 是一个构建大模型应用的框架", metadata={"source": "tech_blog"}),
    Document(page_content="向量数据库用于存储嵌入向量", metadata={"source": "tech_blog"}),
]
vector_store.add_documents(documents=docs, ids=["id1", "id2"])

similar_docs = vector_store.similarity_search("langchain")
print(f"similar_docs: {similar_docs}")

# 注意参数filter类型：filter: dict[str, str] | None = None,
similar_docs_with_filter = vector_store.similarity_search(
    "langchain",
    k=3,
    filter={"source": "tech_blog"},
)
print(f"similar_docs_with_filter: {similar_docs_with_filter}")
