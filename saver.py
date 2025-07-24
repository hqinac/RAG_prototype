from langgraph.graph import Graph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, character
from langchain_experimental.text_splitter import SemanticChunker
import os

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

URI = os.getenv("URI")

def save_vectorstore(documents: list, chunks, size):
    headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
    ]

    character_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
    )
    semantic_splitter = SemanticChunker(   
        embeddings=embeddings,
        buffer_size=1,
        add_start_index=True,
        min_chunk_size= 200
    )
    db = FAISS.from_documents(state["documents"], embeddings)
    
    db=None
    if check_db_exist():
        db = FAISS.load_local(URI, embeddings)
        ndb = FAISS.from_documents(documents, embeddings)
        db.merge_from(ndb)
    else:
        db = FAISS.from_documents(documents, embeddings)
    db.save_local(URI)
    return {"output": "文档已保存"}

def check_db_exist():
    return os.path.exists(URI)
