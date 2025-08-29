from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from rerankers import Document as RerankerDocument
from cache_manager import get_llm, get_faiss, get_bm25, get_doc_cache, get_reranker

import asyncio

load_dotenv()

async def retrieve(strategy, query, filters):
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
            raise ValueError("向量索引不存在，请先上传文档")
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
        hyde_template = """请用最直接，最好的方式回答用中文以下问题，你的问题将作为在数据库中匹配搜索的实例文档。
                        注意：为了提高搜索到的答案与原本问题的关联性，你的回答应该尽可能包含并多提到原问题中的关键词。你的回答应该使用中文。
                问题: {query}
                答案:"""
        eng_hyde_template = """请用最直接，最好的方式用英语回答以下问题，你的问题将作为在数据库中匹配搜索的实例文档。
                        注意：为了提高搜索到的答案与原本问题的关联性，你的回答应该尽可能包含并多提到原问题中的关键词。你的回答应该使用英语。
                问题: {query}
                答案:"""
        prompt_hyde = ChatPromptTemplate.from_template(hyde_template)
        prompt_eng_hyde = ChatPromptTemplate.from_template(eng_hyde_template)
        cn_template = """请用中文翻译提到的问题，不要改变问题原意。如果问题本来就是中文，直接输出原文。
        问题: {query}
        答案:"""
        eng_template = """请用英语翻译提到的问题，不要改变问题原意。如果问题本来就是英语，直接输出原文。
        问题: {query}
        答案:"""
        prompt_cn = ChatPromptTemplate.from_template(cn_template)
        prompt_eng = ChatPromptTemplate.from_template(eng_template)
        cn_chain = prompt_cn | get_llm() | StrOutputParser()
        eng_chain = prompt_eng | get_llm() | StrOutputParser()
        hydechain = (
                prompt_hyde | get_llm() | StrOutputParser()     
                )
        eng_hydechain = (
                prompt_eng_hyde | get_llm() | StrOutputParser()     
                )
        # 选择策略
        match strategy:
            case "hyde":
                cn_query = await cn_chain.ainvoke({"query": query})
                eng_query = await eng_chain.ainvoke({"query": query})
                docs = (await retriever.ainvoke(cn_query))[:5]
                eng_docs = (await retriever.ainvoke(eng_query))[:5]
                docs.extend(eng_docs)
                Strategy = "虚拟文档检索"
            case "bm25rerank":
                # 并行执行FAISS和BM25检索
                cn_query = await cn_chain.ainvoke({"query": query})
                eng_query = await eng_chain.ainvoke({"query": query})
                faiss_docs, faiss_eng_docs, bm25_docs, bm25_eng_docs = await asyncio.gather(
                    retriever.ainvoke(cn_query),
                    retriever.ainvoke(eng_query),
                    bm25.ainvoke(cn_query),
                    bm25.ainvoke(eng_query)
                )
                # 合并结果
                docs = faiss_docs + faiss_eng_docs + bm25_docs + bm25_eng_docs
                # 转换为rerankers库期望的Document格式
                reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i, metadata=doc.metadata) for i, doc in enumerate(docs)]
                results = await reranker.rank_async(query=query, docs=reranker_docs)
                rerankresult = results.top_k(10)
                docs = [Document(page_content=result.text, metadata=result.metadata) for result in rerankresult]
                Strategy = "与bm25结合进行重排序"
            case "faissbert":
                cn_query = await cn_chain.ainvoke({"query": query})
                eng_query = await eng_chain.ainvoke({"query": query})
                docs = (await retriever.ainvoke(cn_query))[:5]
                eng_docs = (await retriever.ainvoke(eng_query))[:5]
                docs.extend(eng_docs)
                Strategy = "faiss向量相似度检索"
            case _:
                #createdquery = await hydechain.ainvoke({"query": query})
                #eng_createdquery = await eng_hydechain.ainvoke({"query": query})
                #print(f"假设文档为：{createdquery}")
                cn_query = await cn_chain.ainvoke({"query": query})
                eng_query = await eng_chain.ainvoke({"query": query})
                faiss_docs, faiss_eng_docs, bm25_docs, bm25_eng_docs = await asyncio.gather(
                    retriever.ainvoke(cn_query),
                    retriever.ainvoke(eng_query),
                    bm25.ainvoke(cn_query),
                    bm25.ainvoke(eng_query)
                )
                # 合并结果
                docs = faiss_docs + faiss_eng_docs + bm25_docs + bm25_eng_docs
                # 转换为rerankers库期望的Document格式
                reranker_docs = [RerankerDocument(text=doc.page_content, doc_id=i, metadata=doc.metadata) for i, doc in enumerate(docs)]
                results = await reranker.rank_async(query=cn_query, docs=reranker_docs)
                rerankresult = results.top_k(10)
                docs = [Document(page_content=result.text, metadata=result.metadata) for result in rerankresult]
                Strategy = "建立虚拟文档，将faiss与bm25检索的结果进行重排序。"
        
        return docs, Strategy
        
    except Exception as e:
        raise ValueError(f"检索过程中出现错误: {str(e)}")

