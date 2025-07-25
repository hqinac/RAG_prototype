from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import os

load_dotenv()
def get_embeddings():
    """延迟初始化embeddings"""
    return DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    )

# 设置默认的向量数据库存储路径
URI = os.getenv("URI", "./vector_db")

def save_vectorstore(documents: list[Document], chunks, size):
    embeddings = get_embeddings()  # 在函数内部初始化embeddings
    headers = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题")
    ]
    existed = []
    embedded = []
    totalsplits = []
    db=None
    if check_db_exist():
        db = FAISS.load_local(URI, embeddings)
    for i, document in enumerate(documents):
        exists = False
        if db:
            exists = bool(db.similarity_search(
                query="任意文本",  # 查询文本不重要，仅触发搜索
                k=1,             # 只返回1个结果
                where={"source": document.metadata["source"]}  # 替换为你的metadata过滤条件
            ))
        if exists:
            existed.append(document.metadata["source"])
            continue
        match chunks[i]:
            case "fixed":
                text_splitter = CharacterTextSplitter(chunk_size = size[i])
                splits = text_splitter.split_documents([document])
            case "recursive":
                text_splitter = RecursiveCharacterTextSplitter(chunk_size = size[i])
                splits = text_splitter.split_documents([document])
            case "md":
                text_splitter = MarkdownHeaderTextSplitter(headers)
                temp_docs = text_splitter.split_documents([document])
                splits = []
                for doc in temp_docs:
                    if len(doc.page_content) > size[i]:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size = size[i])
                        splits.extend(text_splitter.split_documents([doc]))
                    else:
                        splits.extend(doc)

            case _:
                text_splitter = SemanticChunker(   
                embeddings=embeddings,
                buffer_size=size[i]/10,
                number_of_chunks= len(document.page_content)//size[i],
                min_chunk_size= size[i]/2,

            )
                splits = text_splitter.split_documents([document])
        embedded.append(document.metadata["source"])
        totalsplits.extend(splits)

    #将切片写入数据库
    if db:
        ndb = FAISS.from_documents(totalsplits, embeddings)
        db.merge_from(ndb)
    else:
        db = FAISS.from_documents(totalsplits, embeddings)
    db.save_local(URI)
    if existed:
        return f"在数据库中能找与{existed}同名的文档，请确认是否已上传过，如果想要强制上传，请为文档改名，其余文档{embedded}已成功保存到数据库中。"
    return f"文档{embedded}已成功保存到数据库中。"

def check_db_exist():
    return os.path.exists(URI)
