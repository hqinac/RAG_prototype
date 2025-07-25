from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain_qwq import ChatQwen
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import os

load_dotenv()

def get_llm():
    """获取LLM实例"""
    return ChatQwen(
        model=os.getenv("DASH_MODEL_NAME"),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_BASE_URL")
    )

def retrieve(strategy,query,embeddings):
    if strategy == "faissbert":
        db = FAISS.load_local("./vector_db", embeddings)
        docs = db.similarity_search(query, k=3)
    if strategy == "hyde":
        docs = hyde(query,embeddings)
    if strategy == "bm25rerank":
        db = BM25VectorDB.from_documents(docs)
        docs = db.similarity_search(query, k=3)
    return docs