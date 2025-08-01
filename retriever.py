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

import os
import pickle


load_dotenv()

def get_llm():
    """获取LLM实例"""
    return ChatQwen(
        model=os.getenv("DASH_MODEL_NAME"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )

async def retrieve(strategy,query,filters,embeddings):
    '''
    初始化检索器
    '''
    URI = os.getenv("URI", "./saved_files")
    vectorstore = FAISS.load_local(f"{URI}/faiss_index", embeddings, allow_dangerous_deserialization=True)
    if not os.path.exists(f"{URI}/split_docs.pkl") or not os.path.exists(f"{URI}/bm25.pkl"):
        raise ValueError("bm25数据库不存在，请先上传文档")
    with open(f"{URI}/split_docs.pkl", "rb") as f:
        slices = pickle.load(f)
    with open(f"{URI}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    
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
    if filters != []:
            filtered = [doc for doc in slices if doc.metadata["source"] in filters]
            bm25 = BM25Retriever.from_documents(filtered)
            bm25.k = 10
    if not bm25:
        raise ValueError("mb25检索器创建失败，请检查原因")
    if not retriever:
        raise ValueError("faiss检索器创建失败，请检查原因")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, retriever], weights=[0.5, 0.5]
        )
    reranker = Reranker(model_name=r'D:\HuggenfaceModel\BAAIbge-reranker-large\bge-reranker-large', lang= 'zh', model_type="cross-encoder")
    '''
    初始化hyde模板
    '''
    hyde_template = """请写一段答案回答以下问题,回答字数应控制在100字以内：
            问题: {query}
            答案:"""
    prompt_hyde = ChatPromptTemplate.from_template(hyde_template)
    '''
    选择策略
    '''
    match strategy:
        case "hyde":
            hydechain = (
            prompt_hyde | get_llm() | StrOutputParser()|retriever 
            )
            docs = await hydechain.ainvoke({"query": query})
            Strategy = "虚拟文档检索"
        case "bm25rerank":
            docs = await retriever.ainvoke(query)
            docs.extend(await bm25.ainvoke(query))
            # 转换为rerankers库期望的Document格式
            reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i,metadata=doc.metadata) for i, doc in enumerate(docs)]
            results = await reranker.rank_async(query=query, docs=reranker_docs)
            rerankresult = results.top_k(10)
            docs = [Document(page_content=result.text,metadata=result.metadata) for result in rerankresult]
            Strategy = "与bm25结合进行重排序"
        case "faissbert":
            docs = await retriever.ainvoke(query)
            Strategy = "faiss向量相似度检索"
        case _:
            querychain = prompt_hyde | get_llm() | StrOutputParser()
            createdquery = await querychain.ainvoke({"query": query})
            docs = await retriever.ainvoke(createdquery)
            docs.extend(await bm25.ainvoke(createdquery))
            # 转换为rerankers库期望的Document格式
            reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i,metadata=doc.metadata) for i, doc in enumerate(docs)]
            results = await reranker.rank_async(query=createdquery, docs=reranker_docs)
            rerankresult = results.top_k(10)
            docs = [Document(page_content=result.text,metadata=result.metadata) for result in rerankresult]
            Strategy = "建立虚拟文档，将faiss与bm25检索的结果进行重排序。"
    return docs,Strategy

