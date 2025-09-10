from langchain_core.documents import Document
import os
import logging
import chromadb
import pickle
from tablerecognizer import mdfile_recognizer
from langchain_community.retrievers import BM25Retriever
from cacheRAG import get_embeddings, get_chroma, get_bm25, get_doc_info_cache, update_bm25_cache, get_local_store

async def save_vectorstore(documents: list[Document], size, useReplace = True):
    """
    保存向量数据库
    
    Args:
        documents: 文档列表
        size: 切片大小
        useReplace: 是否替换已存在的文档,不替换则将两种切片同时加入向量库
    
    Returns:
        str: 保存结果信息
    """
    '''
    # 检查documents参数的有效性
    if not documents:
        return "错误：没有文档可处理", []
    
    # 检查documents中的每个元素是否为有效的Document对象
    for i, document in enumerate(documents):
        if not hasattr(document, 'page_content') or not hasattr(document, 'metadata'):
            return f"错误：文档索引 {i} 不是有效的Document对象，类型为: {type(document)}", []
    '''
    
    import time
    start_time = time.time()
    
    embeddings = get_embeddings()  # 在函数内部初始化embeddings
    
    logging.info("开始加载现有数据库...")
    load_start = time.time()
    
    # 在函数开头就获取所有缓存数据库
    chroma_db = None
    doc_info = None
    bm25 = None
    
    chroma_db = get_chroma()
    bm25 = get_bm25()

    all_metadata = chroma_db.get(include=["metadatas"])["metadatas"]  
    unique_filenames = {meta.get("source") for meta in all_metadata if meta.get("source")}
    doc_info = list(unique_filenames)
    
    load_time = time.time() - load_start
    logging.info(f"数据库加载完成，耗时: {load_time:.2f}秒")
    
    logging.info("开始文档切片处理...")
    split_start = time.time()
    
    existed = []
    embedded = []
    embedded_chunk = []
    embedded_size = []
    totalsplits = []
    
    # 对每个文档分别处理
    for i, document in enumerate(documents):
            
        # 检查文档是否已存在
        exists = bool(document.metadata["source"] in doc_info)
        if exists:
            existed.append(document.metadata["source"])
            print(f"文件已上传过：{document.metadata['source']}")
        else: 
            embedded.append(document.metadata["source"])

        # 根据切片方式处理文档
        # 默认使用自定义结构切片
        #print(f"使用结构切片方式处理文档({len(document.page_content)}字符)")
        splits = mdfile_recognizer(document, chunk_size=size)
        
        # 确保每个切片都保留原始文档的source元数据
        for j, split in enumerate(splits):
            if "source" not in split.metadata or not split.metadata["source"]:
                split.metadata["source"] = document.metadata.get("source", "未知来源")
            split.metadata["chunk_id"] = j
        
        totalsplits.extend(splits)
    
    split_time = time.time() - split_start
    logging.info(f"文档切片处理完成，共生成 {len(totalsplits)} 个切片，耗时: {split_time:.2f}秒")
    
    '''
    # 为所有切片添加ID
    for j, split in enumerate(totalsplits):
        split.metadata["chunk_id"] = j
    '''
    
   
    # 如果没有新文档需要处理，直接返回
    if not totalsplits:
        total_time = time.time() - start_time
        logging.info(f"总耗时: {total_time:.2f}秒")
        return "没有新文档需要处理。", embedded

    logging.info("开始向量数据库构建...")
    vector_start = time.time()

    def bm25tokenizer(text):
        if text.isalnum():
            return text.split()
        return list(jieba.cut(text))
    
    # 性能优化：优先使用增量更新而非重建
    if existed == [] or not useReplace:
        # 没有重复文档，使用增量更新（最快）
        logging.info("使用增量更新模式...")
        # 批量计算embedding以提高性能
        if totalsplits:
            chroma_db.add(
                documents = [doc.page_content for doc in totalsplits],
                metadatas = [doc.metadata for doc in totalsplits],
                ids = [str(uuid.uuid4()) for _ in totalsplits]
            )

        all_splits = chroma_db.get(include=["documents", "metadatas"])
        all_splits = [Document(page_content=document, metadata=met) for document, met in zip(all_splits["documents"], all_splits['metadatas'])]
        bm25 = BM25Retriever.from_documents(all_splits, k = 10, preprocess_function=bm25tokenizer)

    else:
        # 有重复文档或首次创建，需要重建
        logging.info("检测到重复文件，将重复文件覆盖...")
        metadata_condition = {
        "source": {"$in": source_list}  
        }
        results = chroma_db.query(
            where=metadata_condition,  
            n_results=10000,  
            include=["ids"]  
            )
    
        # 提取符合条件的文档ID列表
        document_ids = results["ids"][0]
    
        if not document_ids:
            print("没有找到符合条件的文档")

        chroma_db.delete(ids=document_ids)
        print(f"成功删除 {len(document_ids)} 个符合条件的文档")

        all_splits = chroma_db.get(include=["documents", "metadatas"])
        all_splits = [Document(page_content=document, metadata=met) for document, met in zip(all_splits["documents"], all_splits['metadatas'])]
        bm25 = BM25Retriever.from_documents(all_splits, k = 10, preprocess_function=bm25tokenizer)

    vector_time = time.time() - vector_start
    print(f"向量数据库构建完成，耗时: {vector_time:.2f}秒")

    # 确保目录存在
    if get_local_store():
        os.makedirs(URI, exist_ok=True)  
        print("开始保存到磁盘...")
        save_start = time.time()
    
        with open(f"{URI}/bm25.pkl", "wb") as f:
            pickle.dump(bm25, f)
    
        save_time = time.time() - save_start
        print(f"磁盘保存完成，耗时: {save_time:.2f}秒")
    
    # 同步更新缓存
    update_bm25_cache(bm25)
    
    total_time = time.time() - start_time
    logging.info(f"总耗时: {total_time:.2f}秒")
    
    if existed and useReplace:
        return f"在数据库中能找到与{existed}同名的文档，已在数据库中以切片大小{size}进行更新。"

    if not useReplace:
        return f"文档{existed + embedded}已成功以切片大小{size}保存到数据库中。"
    
    return f"文档{embedded}已成功以切片大小{size}保存到数据库中。"
