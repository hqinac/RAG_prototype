import os
import asyncio
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qwq import ChatQwen
from dotenv import load_dotenv
import pickle
from cache_manager import get_embeddings, get_faiss, get_bm25, get_doc_cache, update_faiss_cache, update_bm25_cache, update_doc_cache

load_dotenv()

# 设置默认的向量数据库存储路径
URI = os.getenv("URI", "./saved_files")

async def save_vectorstore(documents: list[Document], chunks, size, doc_info, language=None):
    """
    保存向量数据库
    
    Args:
        documents: 文档列表
        chunks: 切片方式列表
        size: 切片大小列表
        doc_info: 文档信息列表（已存在的文件名）
        language: 使用的语言（可选参数）
    
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
    db = None
    if check_db_exist():
        # 优先从缓存中获取FAISS数据库
        db = get_faiss()
        if db is None:
            # 如果缓存中没有，则从磁盘加载
            db = FAISS.load_local(f"{URI}/faiss_index", embeddings, allow_dangerous_deserialization=True)
            # 加载后更新缓存
            update_faiss_cache(db)
    
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
        
        # 根据语言设置分隔符和正则表达式
        current_language = language[i] if language and i < len(language) else "cn"
        
        # 英文分隔符配置
        if current_language == "en":
            separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
            regex_pattern = r'[.!?;,\n]+'
        # 中文分隔符配置（默认）
        else:  # cn
            separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            regex_pattern = r'[。！？；，\n]+'
            
        match chunks[i]:
            case "fixed":
                text_splitter = CharacterTextSplitter(
                    chunk_size=size[i],
                    separator=separators[0] if separators else "\n\n"
                )
                splits =await text_splitter.atransform_documents([document])
            case "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size[i],
                    separators=separators
                )
                splits = await text_splitter.atransform_documents([document])
            case "md":
                text_splitter = MarkdownTextSplitter(headers)
                temp_docs = await text_splitter.atransform_documents([document])
                splits = []
                for doc in temp_docs:
                    if len(doc.page_content) > size[i]:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=size[i],
                            separators=separators
                        )
                        splits.extend(await text_splitter.atransform_documents([doc]))
                    else:
                        splits.append(doc)

            case _:
                text_splitter = SemanticChunker(   
                embeddings=embeddings,
                buffer_size=size[i]/10,
                number_of_chunks= len(document.page_content)//size[i],
                sentence_split_regex= regex_pattern,
                min_chunk_size= size[i]/2,

            )
                splits = await text_splitter.atransform_documents([document])
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
        
        # 优先从缓存中获取文档列表
        docs = get_doc_cache()
        if docs is None:
            # 如果缓存中没有，则从磁盘加载
            with open(f"{URI}/split_docs.pkl", "rb") as f:
                docs = pickle.load(f)
            # 加载后更新缓存
            update_doc_cache(docs)
        else:
            # 如果从缓存获取，需要复制一份以避免修改原缓存
            docs = docs.copy()
        
        docs.extend(totalsplits)
        
        # 优先从缓存中获取BM25检索器
        bm25 = get_bm25()
        if bm25 is None:
            # 如果缓存中没有，则从磁盘加载
            with open(f"{URI}/bm25.pkl", "rb") as f:
                bm25 = pickle.load(f)
            # 加载后更新缓存
            update_bm25_cache(bm25)
        
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
