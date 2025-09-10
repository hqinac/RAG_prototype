"""统一缓存管理器
集中管理所有模型和检索器的缓存，确保整个graph都能高效访问
"""
import os
import jieba
import pickle
import logging
from typing import Optional
from dotenv import load_dotenv
import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_qwq import ChatQwen
from rerankers import Reranker

load_dotenv()

        
class CacheManager:
    """全局缓存管理器"""
    
    def __init__(self):
        # 模型缓存
        self._llm_cache: Optional[ChatQwen] = None
        self._embeddings_cache: Optional[DashScopeEmbeddings] = None
        
        # 检索器缓存
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._chroma_cache: Optional[chromadb.Collection] = None
        #self._faiss_cache: Optional[FAISS] = None
        self._bm25_cache: Optional[BM25Retriever] = None
        self._reranker_cache: Optional[Reranker] = None
        
        # 文档缓存
        self._doc_info_cache: Optional[list] = None
        
        # 缓存状态标记
        self._cache_loaded = False
        self._useLocalStore = os.getenv("USE_LOCAL_STORE", "True").lower() == "true"
        

        
    def get_llm(self) -> ChatQwen:
        """获取LLM实例（缓存版本）"""
        if self._llm_cache is None:
            try:
                self._llm_cache = ChatQwen(
                    model=os.getenv("DASH_MODEL_NAME"),
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url=os.getenv("DASHSCOPE_BASE_URL"),

                    timeout=os.getenv("DASHSCOPE_API_TIMEOUT", 60)
                )
            except Exception as e:
                print(f"LLM初始化失败: {e}")
                raise
        return self._llm_cache
    
    def get_embeddings(self) -> DashScopeEmbeddings:
        """获取embeddings实例（缓存版本）"""
        if self._embeddings_cache is None:
            self._embeddings_cache = DashScopeEmbeddings(
                model=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
        return self._embeddings_cache
    
    def get_chroma(self) -> Optional[chromadb.Collection]:
        """获取Chroma索引（缓存版本）"""
        if self._chroma_cache is None :
            if self._useLocalStore:
                URI = os.getenv("URI", "./saved_files")
                self._chroma_client = chromadb.PersistentClient(path=f"{URI}/chroma_db")
                self._chroma_cache = self._chroma_client.get_or_create_collection(
                    name="chroma_with_dashscope",
                    embedding_function=self.get_embeddings(),  # 直接传入LangChain的嵌入模型
                )
            else:
                self._chroma_client = chromadb.Client()
                self._chroma_cache = self._chroma_client.get_or_create_collection(
                    name=os.getenv("COLLECTION_NAME", "chroma_with_dashscope"),
                    embedding_function=self.get_embeddings(),  # 直接传入LangChain的嵌入模型
                )
        return self._chroma_cache
    
    def get_bm25(self) -> Optional[BM25Retriever]:
        """获取BM25检索器（缓存版本）"""

        if self._bm25_cache is None:
            if self._useLocalStore:
                URI = os.getenv("URI", "./saved_files")
                if not os.path.exists(f"{URI}/bm25.pkl"):
                    self._bm25_cache = None
                with open(f"{URI}/bm25.pkl", "rb") as f:
                    self._bm25_cache = pickle.load(f)
        return self._bm25_cache
    
    '''
    def get_doc_info_cache(self) -> Optional[list]:
        """获取文档信息缓存（缓存版本）"""
        if self._doc_info_cache is None and self._useLocalCache:
            URI = os.getenv("URI", "./saved_files")
            if not os.path.exists(f"{URI}/doc_info.json"):
                return None
            with open(self.doc_info_file, 'r', encoding='utf-8') as f:
                    self._doc_info_cache = json.load(f)
        return self._doc_info_cache
    '''
    
    def get_reranker(self) -> Reranker:
        """获取Reranker实例（缓存版本）"""
        if self._reranker_cache is None:
            saved_reranker = os.getenv("RERANKER_NAME_OR_PATH", "BAAI/bge-reranker-large")
            self._reranker_cache = Reranker(
                model_name=saved_reranker,
                lang='zh',
                model_type="cross-encoder"
            )
        return self._reranker_cache
    
    def get_local_store(self) -> bool:
        """获取是否使用本地存储"""
        return self._useLocalStore
    
    def clear_retriever_cache(self):
        """清除检索器相关缓存（用于数据更新后重新加载）"""
        self._chroma_cache = None
        self._doc_info_cache = None
        self._cache_loaded = False
    
    def clear_all_cache(self):
        """清除所有缓存"""
        self._llm_cache = None
        self._embeddings_cache = None
        self._chroma_cache = None
        self._doc_info_cache = None
        self._reranker_cache = None
        self._cache_loaded = False
    
    def update_chroma_cache(self, chroma_instance: chromadb.Collection):
        """更新Chroma缓存"""
        self._chroma_cache = chroma_instance
    
    def update_bm25_cache(self, bm25_instance: BM25Retriever):
        """更新BM25缓存"""
        self._bm25_cache = bm25_instance
    
    def update_doc_cache(self, doc_list: list):
        """更新文档缓存"""
        self._doc_cache = doc_list
    '''
    def update_doc_info_cache(self, doc_info: list):
        """更新文档信息缓存"""
        self._doc_info_cache = doc_info
    '''
    
    def is_cache_loaded(self) -> bool:
        """检查缓存是否已加载"""
        return self._cache_loaded
    
    def preload_all(self):
        """预加载所有缓存（可选的性能优化）"""
        try:
            self.get_llm()
            self.get_embeddings()
            self.get_chroma()
            #self.get_doc_info_cache()
            self.get_reranker()
            self._cache_loaded = True
            logging.info("所有缓存预加载完成")
        except Exception as e:
            print(f"缓存预加载失败: {e}")
            
# 全局缓存管理器实例
cache_manager = CacheManager()

# 提供便捷的访问函数
def get_llm():
    """获取LLM实例"""
    return cache_manager.get_llm()

def get_embeddings():
    """获取embeddings实例"""
    return cache_manager.get_embeddings()

def get_chroma() -> Optional[chromadb.Collection]:
    """获取Chroma索引"""
    return cache_manager.get_chroma()

def get_bm25() -> Optional[BM25Retriever]:
    """获取BM25检索器"""
    return cache_manager.get_bm25()

def get_doc_info_cache() -> Optional[list]:
    """获取文档信息缓存"""
    return cache_manager.get_doc_info_cache()

def get_reranker():
    """获取Reranker实例"""
    return cache_manager.get_reranker()

def clear_retriever_cache():
    """清除检索器缓存"""
    cache_manager.clear_retriever_cache()

def clear_all_cache():
    """清除所有缓存"""
    cache_manager.clear_all_cache()

def preload_all_cache():
    """预加载所有缓存"""
    cache_manager.preload_all()

'''
def update_faiss_cache(faiss_instance: FAISS):
    """更新FAISS缓存"""
    cache_manager.update_faiss_cache(faiss_instance)
'''
def update_bm25_cache(bm25_instance: BM25Retriever):
    """更新BM25缓存"""
    cache_manager.update_bm25_cache(bm25_instance)

def get_local_store() -> bool:
    """获取是否使用本地存储"""
    return cache_manager.get_local_store()

'''
def update_doc_info_cache(doc_info: list):
    """更新文档信息缓存"""
    cache_manager.update_doc_cache(doc_list)
'''
