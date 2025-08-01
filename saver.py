import os
import asyncio
from re import A
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
    import time
    start_time = time.time()
    
    embeddings = get_embeddings()  # 在函数内部初始化embeddings
    
    print("开始加载现有数据库...")
    load_start = time.time()
    
    # 在函数开头就获取所有缓存数据库
    db = None
    docs = None
    bm25 = None
    
    if check_db_exist():
        # 优先从缓存中获取FAISS数据库
        db = get_faiss()
        if db is None:
            # 如果缓存中没有，则从磁盘加载
            db = FAISS.load_local(f"{URI}/faiss_index", embeddings, allow_dangerous_deserialization=True)
            # 加载后更新缓存
            update_faiss_cache(db)
        
        # 优先从缓存中获取文档列表
        docs = get_doc_cache()
        if docs is None:
            # 如果缓存中没有，则从磁盘加载
            with open(f"{URI}/split_docs.pkl", "rb") as f:
                docs = pickle.load(f)
            # 加载后更新缓存
            update_doc_cache(docs)
        
        # 优先从缓存中获取BM25检索器
        bm25 = get_bm25()
        if bm25 is None:
            # 如果缓存中没有，则从磁盘加载
            with open(f"{URI}/bm25.pkl", "rb") as f:
                bm25 = pickle.load(f)
            # 加载后更新缓存
            update_bm25_cache(bm25)
    
    load_time = time.time() - load_start
    print(f"数据库加载完成，耗时: {load_time:.2f}秒")
    
    print("开始文档切片处理...")
    split_start = time.time()
    
    headers = [
        ("#", "一级标题"),
        ("##", "二级标题"),
        ("###", "三级标题")
    ]
    existed = []
    embedded = []
    totalsplits = []
    
    # 对每个文档分别处理
    for i, document in enumerate(documents):
        # 检查索引范围，避免越界
        if i >= len(chunks) or i >= len(size):
            print(f"警告：文档索引 {i} 超出 chunks 或 size 列表范围，跳过处理")
            break
            
        # 检查文档是否已存在
        exists = False
        if db:
            exists = document.metadata["source"] in doc_info
        if exists:
            existed.append(document.metadata["source"])
            print(f"文件已存在：{document.metadata['source']}")
        else: 
            embedded.append(document.metadata["source"])
        
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
            
        # 根据切片方式处理文档
        match chunks[i]:
            case "fixed":
                text_splitter = CharacterTextSplitter(
                    chunk_size=size[i],
                    separator=separators[0] if separators else "\n\n"
                )
                splits = await text_splitter.atransform_documents([document])
            case "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size[i],
                    separators=separators
                )
                splits = await text_splitter.atransform_documents([document])
            case "document":
                text_splitter = MarkdownTextSplitter()
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
                # 默认使用语义切片
                print(f"使用语义切片处理文档({len(document.page_content)}字符)")
                text_splitter = SemanticChunker(   
                    embeddings=embeddings,
                    buffer_size=max(10, int(size[i]/15)),
                    number_of_chunks=max(3, int(len(document.page_content)//int(size[i]*0.8))),
                    sentence_split_regex=regex_pattern,
                    min_chunk_size=max(50, int(size[i]/5)),
                )
                splits = await text_splitter.atransform_documents([document])
        totalsplits.extend(splits)
    
    split_time = time.time() - split_start
    print(f"文档切片处理完成，共生成 {len(totalsplits)} 个切片，耗时: {split_time:.2f}秒")
    
    # 为所有切片添加ID
    for j, split in enumerate(totalsplits):
        split.metadata["id"] = j
    
   
    # 如果没有新文档需要处理，直接返回
    if not totalsplits:
        total_time = time.time() - start_time
        print(f"总耗时: {total_time:.2f}秒")
        return "没有新文档需要处理。", embedded

    print("开始向量数据库构建...")
    vector_start = time.time()


    tempdocs = docs.copy() if docs else []
    
    # 性能优化：优先使用增量更新而非重建
    if db and existed == []:
        # 没有重复文档，使用增量更新（最快）
        print("使用增量更新模式...")
        
        # 批量计算embedding以提高性能
        if totalsplits:
            texts = [doc.page_content for doc in totalsplits]
            embeddings_vectors = embeddings.embed_documents(texts)
            
            # 使用预计算的embedding创建FAISS
            import numpy as np
            vectors_array = np.array(embeddings_vectors).astype('float32')
            ndb = FAISS.from_embeddings(
                list(zip(texts, vectors_array)), 
                embeddings
            )
            
            # 更新文档元数据
            for i, doc in enumerate(totalsplits):
                ndb.docstore.add({str(i): doc})
            
            # 合并到现有数据库
            db.merge_from(ndb)
            
            # 增量更新BM25
            for split in totalsplits:
                bm25.add_document(split)
    else:
        # 有重复文档或首次创建，需要重建
        print("使用重建模式...")
        tempdocs = [doc for doc in docs if doc.metadata["source"] not in existed] if existed != [] else tempdocs
        tempdocs.extend(totalsplits)
        
        # 批量处理embedding以提高性能
        if tempdocs:
            print(f"批量处理 {len(tempdocs)} 个文档切片的embedding...")
            texts = [doc.page_content for doc in tempdocs]
            
            # 分批处理大量文档以避免内存问题
            batch_size = 100
            if len(texts) > batch_size:
                print(f"分批处理，每批 {batch_size} 个文档...")
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = embeddings.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 个文档")
            else:
                all_embeddings = embeddings.embed_documents(texts)
            
            # 使用预计算的embedding创建FAISS
            import numpy as np
            vectors_array = np.array(all_embeddings).astype('float32')
            db = FAISS.from_embeddings(
                list(zip(texts, vectors_array)), 
                embeddings
            )
            
            # 更新文档元数据
            for i, doc in enumerate(tempdocs):
                db.docstore.add({str(i): doc})
        else:
            db = FAISS.from_documents(tempdocs, embeddings)
            
        bm25 = BM25Retriever.from_documents(tempdocs)
        bm25.k = 10

    # 确保目录存在
    os.makedirs(URI, exist_ok=True)
    
    vector_time = time.time() - vector_start
    print(f"向量数据库构建完成，耗时: {vector_time:.2f}秒")
    
    print("开始保存到磁盘...")
    save_start = time.time()
    
    # 保存到磁盘
    db.save_local(f"{URI}/faiss_index")
    with open(f"{URI}/split_docs.pkl", "wb") as f:
        pickle.dump(tempdocs, f)
    with open(f"{URI}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    
    save_time = time.time() - save_start
    print(f"磁盘保存完成，耗时: {save_time:.2f}秒")
    
    # 同步更新缓存
    update_faiss_cache(db)
    update_bm25_cache(bm25)
    update_doc_cache(tempdocs)
    
    total_time = time.time() - start_time
    print(f"=== 性能统计 ===")
    print(f"数据库加载: {load_time:.2f}秒")
    print(f"文档切片: {split_time:.2f}秒")
    print(f"向量构建: {vector_time:.2f}秒")
    print(f"磁盘保存: {save_time:.2f}秒")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"===============")
    
    if existed:
        return f"在数据库中能找到与{existed}同名的文档，已在数据库中进行更新，如果想要两份文档同时存在，请为数据库中原有的文档改名，其余文档{embedded}已成功以切片方式{chunks}保存到数据库中。", embedded

    return f"文档{embedded}已成功以切片方式{chunks}保存到数据库中。", embedded

def check_db_exist():
    return (os.path.exists(URI) and 
            os.path.exists(f"{URI}/faiss_index") and
            os.path.exists(f"{URI}/split_docs.pkl") and
            os.path.exists(f"{URI}/bm25.pkl"))
