from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain_qwq import ChatQwen
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_community.document_compressors import FlashrankRerank
from rerankers import Reranker
from rerankers import Document as RerankerDocument
from typing import Any
from cache_manager import get_llm, get_faiss, get_bm25, get_doc_cache, get_reranker

import os
import pickle
import asyncio

load_dotenv()

async def retrieve(strategy, query, filters, embeddings):
    '''
    使用缓存的检索器进行检索
    '''
    try:
        # 使用缓存的组件
        vectorstore = get_faiss()
        slices = get_doc_cache()
        bm25 = get_bm25()
        
        # 检查数据库是否存在
        if vectorstore is None:
            raise ValueError("FAISS索引不存在，请先上传文档")
        if slices is None:
            raise ValueError("文档缓存不存在，请先上传文档")
        if bm25 is None:
            raise ValueError("BM25数据库不存在，请先上传文档")
        
        # 创建检索器
        if filters == []:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 10}   
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"filter": {"source": {"$in": filters}},
                                "k": 10}   
            )
        
        # 如果有过滤条件，创建过滤后的BM25检索器
        if filters != []:
            filtered = [doc for doc in slices if doc.metadata["source"] in filters]
            bm25 = BM25Retriever.from_documents(filtered)
            bm25.k = 10
        
        if not bm25:
            raise ValueError("BM25检索器创建失败，请检查原因")
        if not retriever:
            raise ValueError("FAISS检索器创建失败，请检查原因")
        
        # 获取缓存的reranker
        reranker = get_reranker()
        
        # 初始化hyde模板
        hyde_template = """请直接回答以下问题，要求：
                1. 只回答问题中明确提到的内容，不要扩展到其他相关话题
                2. 保持答案简洁，控制在50字以内
                3. 如果问题提到具体疾病名称，答案中必须包含该疾病名称
                问题: {query}
                答案:"""
        prompt_hyde = ChatPromptTemplate.from_template(hyde_template)
        
        # 选择策略
        match strategy:
            case "hyde":
                hydechain = (
                    prompt_hyde | get_llm() | StrOutputParser() | retriever 
                )
                docs = await hydechain.ainvoke({"query": query})
                Strategy = "虚拟文档检索"
            case "bm25rerank":
                # 并行执行FAISS和BM25检索
                faiss_docs, bm25_docs = await asyncio.gather(
                    retriever.ainvoke(query),
                    bm25.ainvoke(query)
                )
                # 合并结果
                docs = faiss_docs + bm25_docs
                # 转换为rerankers库期望的Document格式
                reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i, metadata=doc.metadata) for i, doc in enumerate(docs)]
                results = await reranker.rank_async(query=query, docs=reranker_docs)
                rerankresult = results.top_k(10)
                docs = [Document(page_content=result.text, metadata=result.metadata) for result in rerankresult]
                Strategy = "与bm25结合进行重排序"
            case "faissbert":
                docs = await retriever.ainvoke(query)
                Strategy = "faiss向量相似度检索"
            case _:
                querychain = prompt_hyde | get_llm() | StrOutputParser()
                createdquery = await querychain.ainvoke({"query": query})
                print(f"假设文档为：{createdquery}")
                faiss_docs, bm25_docs = await asyncio.gather(
                    retriever.ainvoke(createdquery),
                    bm25.ainvoke(createdquery)
                )
                # 合并结果
                docs = faiss_docs + bm25_docs
                # 转换为rerankers库期望的Document格式
                reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i, metadata=doc.metadata) for i, doc in enumerate(docs)]
                results = await reranker.rank_async(query=createdquery, docs=reranker_docs)
                rerankresult = results.top_k(10)
                docs = [Document(page_content=result.text, metadata=result.metadata) for result in rerankresult]
                Strategy = "建立虚拟文档，将faiss与bm25检索的结果进行重排序。"
        
        return docs, Strategy
        
    except Exception as e:
        raise ValueError(f"检索过程中出现错误: {str(e)}")

