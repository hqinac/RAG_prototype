import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from dbsaver import save_vectorstore
import chromadb
from cacheRAG import get_chroma, get_local_store

splitterapi = FastAPI(title="数据库处理api")

class SplitResponse(BaseModel):
    """文件分割处理的响应模型"""
    embedding_result: str = Field(description="嵌入处理结果的详细描述") 
    collection_features: Dict[str, Any] = Field(description="数据库的详细特征信息，包含ChromaDB集合的名称、ID、文档数量和客户端存储路径 与 bm25索引的存储路径") # 集合特征信息
    warnings: List[str] = Field(description="文档读取过程中的警告信息列表")  # 处理过程中的警告信息
    
    class Config:
        json_schema_extra = {
            "example": {
                "embedding_result": "成功处理文档",
                "collection_features": {
                    "collection name": "default",
                    "collection id": "abc123",
                    "collection count": 10,
                    "client path": "path/to/root/directory/saved_files/chromadb",
                    "bm25 path": "path/to/root/directory/saved_files/bm25.pkl"
                },
                "warnings": []
            }
        }

# 安全设置：限制可以访问的根目录，防止路径遍历攻击
#SAFE_ROOT_DIRECTORY = os.path.abspath("./allowed_files")

# 确保安全目录存在
#os.makedirs(SAFE_ROOT_DIRECTORY, exist_ok=True)


@splitterapi.post("/split", summary="将文件解析为数据库", 
description="""
    将指定文件夹下的Markdown文件解析并存储到向量数据库中。
    
    **功能说明：**
    - 扫描指定文件夹下的所有.md文件
    - 将文件内容分割成指定大小的文本块
    - 生成向量嵌入并存储到ChromaDB
    - 生成bm25索引
    - 如果.env中USE_LOCAL_CACHE=True(默认为true), 则会将数据库存储到本地目录并返回绝对路径,否则只将数据库加入缓存, 在程序退出时清空。
    - 可选择是否替换重复上传的文档

    **参数说明：**
    
    - **file_path**: 包含Markdown文件的文件夹路径
      - 必须是有效的目录路径
      - 目录下应包含.md格式的文件
      - 支持相对路径和绝对路径
    
    - **chunk_size**: 文本分块大小（默认500字符）
    
    - **useReplace**: 数据处理模式（默认True）
      - True：文件夹中文件与数据库中已有文件相同时，完全替换现有数据库内容
      - False：将同一文件的新切片追加到现有数据库
    """,
    response_model=SplitResponse)
async def split_files(file_path: str, chunk_size: int = 500, useReplace: bool = True) -> SplitResponse:
    full_path = os.path.abspath(file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"文件夹不存在：{full_path}")
    
    if not os.path.isdir(full_path):
        raise HTTPException(status_code=400, detail=f"不是一个有效的文件夹：{full_path}")
    
    md_files = []
    file_names = []
    warnings = []
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.lower().endswith('.md'):
                current_file_path = os.path.join(root, file)
                loader = TextLoader(current_file_path, encoding='utf-8')
                documents = loader.load()
                
            
                # 检查是否成功加载文档
                if not documents:
                    warnings.append(f"文件解析失败：{file_path}，跳过此文件。")
                    continue
            
                # 检查第一个文档是否有效
                if not hasattr(documents[0], 'page_content') or not hasattr(documents[0], 'metadata'):
                    warnings.append(f"文档{current_file_path}解析出的对象无效，类型为: {type(documents[0])}，跳过此文件。")
                    continue
            
                documents[0].metadata['source'] = file
                documents[0].metadata['file_path'] = current_file_path
                # 添加到文档列表
                md_files.extend(documents)
                file_names.append(file)

    if md_files == []:
        raise HTTPException(status_code=500, detail=f"文件夹下没有md文件或解析出错：{full_path}，警告信息：{warnings}")
    
    result, = await save_vectorstore(md_files, chunk_size, useReplace)
    collection = get_chroma()
    if get_local_store():
        URI = os.getenv("URI", "./saved_files")
        client_path = os.path.abspath(collection._client._settings.persist_directory)
        bm25_path = os.path.abspath(f"{URI}/bm25.pkl")
    else:
        client_path = None
        bm25_path = None
    collection_features = {"collection name":collection.name, "collection id":collection.id, "collection count":collection.count(), "client path":client_path, "bm25 path":bm25_path}
    return SplitResponse(
        embedding_result=result,
        collection_features=collection_features,
        warnings=warnings
    )

@splitterapi.post("/retrieve", summary="从数据库中检索文档", 
description="""
    从向量数据库中检索与查询相关的文档。
    
    **功能说明：**
    - 基于查询文本生成向量嵌入
    - 在向量数据库中查找最相似的文档
    - 返回检索到的文档内容和元数据
    
    **参数说明：**
    
    - **query**: 检索查询文本
      - 必须是有效的文本字符串
      - 用于生成向量嵌入以进行相似度匹配
    
    - **top_k**: 返回结果数量（默认3）
      - 范围：1-10
      - 控制返回的文档数量
      - 数值越大，返回的文档越丰富，但可能包含重复内容
    """,
    response_model=RetrieveResponse)
async def retrieve_files(query: str, top_k: int = 5, method: str = "bm25rerank") -> RetrieveResponse:
    collection = get_chroma()
    pass