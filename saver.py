import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import pickle
from cache_manager import get_embeddings, update_faiss_cache, update_bm25_cache, update_doc_cache

load_dotenv()

# 设置默认的向量数据库存储路径
URI = os.getenv("URI", "./saved_files")

def save_vectorstore(documents: list[Document], chunks, size, doc_info):
    """
    保存向量数据库
    
    Args:
        documents: 文档列表
        chunks: 切片方式列表
        size: 切片大小列表
        doc_info: 文档信息列表（已存在的文件名）
    
    Returns:
        str: 保存结果信息
    """
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
        db = FAISS.load_local(f"{URI}/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    for i, document in enumerate(documents):
        # 检查索引范围，避免越界
        if i >= len(chunks) or i >= len(size):
            print(f"警告：文档索引 {i} 超出 chunks 或 size 列表范围，跳过处理")
            break
            
        exists = False
        if db:
            exists = bool(document.metadata["source"] in doc_info)
        if exists:
            existed.append(document.metadata["source"])
            print(f"文件已存在：{document.metadata['source']}")
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
                        splits.append(doc)

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
    for j,split in enumerate(totalsplits):
        split.metadata["id"] = j

    # 如果所有文件都已存在，直接返回
    if not embedded and existed:
        return f"在数据库中能找到与{existed}同名的文档，请确认是否已上传过，如果想要强制上传，请为文档改名。", []
    
    # 如果没有新文档需要处理，直接返回
    if not totalsplits:
        return "没有新文档需要处理。", embedded

    # 直接存储整个BM25Retriever对象
    #将切片写入数据库
    if db:
        ndb = FAISS.from_documents(totalsplits, embeddings)
        with open(f"{URI}/split_docs.pkl", "rb") as f:
            docs = pickle.load(f)
        docs.extend(totalsplits)
        with open(f"{URI}/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        for split in totalsplits:
            bm25.add_document(split)
        db.merge_from(ndb)
    else:
        db = FAISS.from_documents(totalsplits, embeddings)
        bm25 = BM25Retriever.from_documents(totalsplits)
        bm25.k = 20
        docs = totalsplits

    # 确保目录存在
    os.makedirs(URI, exist_ok=True)
    
    db.save_local(f"{URI}/faiss_index")
    with open(f"{URI}/split_docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(f"{URI}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    
    # 同步更新缓存
    update_faiss_cache(db)
    update_bm25_cache(bm25)
    update_doc_cache(docs)
    
    if existed:
        return f"在数据库中能找与{existed}同名的文档，请确认是否已上传过，如果想要强制上传，请为文档改名，其余文档{embedded}已成功以切片方式{chunks}保存到数据库中。",embedded

    return f"文档{embedded}已成功以切片方式{chunks}保存到数据库中。",embedded

def check_db_exist():
    return (os.path.exists(URI) and 
            os.path.exists(f"{URI}/faiss_index") and
            os.path.exists(f"{URI}/split_docs.pkl") and
            os.path.exists(f"{URI}/bm25.pkl"))
