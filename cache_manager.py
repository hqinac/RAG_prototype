"""
统一缓存管理器
集中管理所有模型和检索器的缓存，确保整个graph都能高效访问
"""
import os
import pickle
from typing import Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
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
        self._faiss_cache: Optional[FAISS] = None
        self._bm25_cache: Optional[BM25Retriever] = None
        self._reranker_cache: Optional[Reranker] = None
        
        # 文档缓存
        self._doc_cache: Optional[list] = None
        
        # 缓存状态标记
        self._cache_loaded = False
        
    def get_llm(self) -> ChatQwen:
        """获取LLM实例（缓存版本）"""
        if self._llm_cache is None:
            self._llm_cache = ChatQwen(
                model=os.getenv("DASH_MODEL_NAME"),
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL")
            )
        return self._llm_cache
    
    def get_embeddings(self) -> DashScopeEmbeddings:
        """获取embeddings实例（缓存版本）"""
        if self._embeddings_cache is None:
            self._embeddings_cache = DashScopeEmbeddings(
                model="text-embedding-v3",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
            )
        return self._embeddings_cache
    
    def get_faiss(self) -> Optional[FAISS]:
        """获取FAISS索引（缓存版本）"""
        if self._faiss_cache is None:
            URI = os.getenv("URI", "./saved_files")
            if not os.path.exists(f"{URI}/faiss_index"):
                return None
            embeddings = self.get_embeddings()
            self._faiss_cache = FAISS.load_local(
                f"{URI}/faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        return self._faiss_cache
    
    def get_bm25(self) -> Optional[BM25Retriever]:
        """获取BM25检索器（缓存版本）"""
        if self._bm25_cache is None:
            URI = os.getenv("URI", "./saved_files")
            if not os.path.exists(f"{URI}/bm25.pkl"):
                return None
            with open(f"{URI}/bm25.pkl", "rb") as f:
                self._bm25_cache = pickle.load(f)
        return self._bm25_cache
    
    def get_doc_cache(self) -> Optional[list]:
        """获取文档缓存（缓存版本）"""
        if self._doc_cache is None:
            URI = os.getenv("URI", "./saved_files")
            if not os.path.exists(f"{URI}/split_docs.pkl"):
                return None
            with open(f"{URI}/split_docs.pkl", "rb") as f:
                self._doc_cache = pickle.load(f)
        return self._doc_cache
    
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
    
    def clear_retriever_cache(self):
        """清除检索器相关缓存（用于数据更新后重新加载）"""
        self._faiss_cache = None
        self._bm25_cache = None
        self._doc_cache = None
        self._cache_loaded = False
    
    def clear_all_cache(self):
        """清除所有缓存"""
        self._llm_cache = None
        self._embeddings_cache = None
        self._faiss_cache = None
        self._bm25_cache = None
        self._reranker_cache = None
        self._doc_cache = None
        self._cache_loaded = False
    
    def update_faiss_cache(self, faiss_instance: FAISS):
        """更新FAISS缓存"""
        self._faiss_cache = faiss_instance
    
    def update_bm25_cache(self, bm25_instance: BM25Retriever):
        """更新BM25缓存"""
        self._bm25_cache = bm25_instance
    
    def update_doc_cache(self, doc_list: list):
        """更新文档缓存"""
        self._doc_cache = doc_list
    
    def is_cache_loaded(self) -> bool:
        """检查缓存是否已加载"""
        return self._cache_loaded
    
    def preload_all(self):
        """预加载所有缓存（可选的性能优化）"""
        try:
            self.get_llm()
            self.get_embeddings()
            self.get_faiss()
            self.get_bm25()
            self.get_doc_cache()
            self.get_reranker()
            self._cache_loaded = True
            print("所有缓存预加载完成")
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

def get_faiss() -> Optional[FAISS]:
    """获取FAISS索引"""
    return cache_manager.get_faiss()

def get_bm25() -> Optional[BM25Retriever]:
    """获取BM25检索器"""
    return cache_manager.get_bm25()

def get_doc_cache() -> Optional[list]:
    """获取文档缓存"""
    return cache_manager.get_doc_cache()

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

def update_faiss_cache(faiss_instance: FAISS):
    """更新FAISS缓存"""
    cache_manager.update_faiss_cache(faiss_instance)

def update_bm25_cache(bm25_instance: BM25Retriever):
    """更新BM25缓存"""
    cache_manager.update_bm25_cache(bm25_instance)

def update_doc_cache(doc_list: list):
    """更新文档缓存"""
    cache_manager.update_doc_cache(doc_list)