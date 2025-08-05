from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np



def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

class RetrievalEvaluator:
    def __init__(self, embedding):
        self.embedding = embedding
    
    def evaluate(self, query, retrieved_docs, top_k=3):
        """评估检索结果质量"""
        # 1. 计算检索结果与query的语义相似度
        query_embed = self.embedding.embed_query(query)
        doc_embeds = [self.embedding.embed_query(doc.page_content) for doc in retrieved_docs]
        similarities = [cosine_similarity(query_embed, doc_embed) for doc_embed in doc_embeds]
        
        # 2. 计算覆盖度分数（基于检索结果多样性）
        diversity_score = self._calculate_diversity(doc_embeds)
        
        # 3. 计算位置衰减分数
        #position_score = sum([(1/i) * sim for i, sim in enumerate(similarities, 1)])
        
        return {
            "avg_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "diversity_score": diversity_score
            #"position_score": position_score
        }
    
    def _calculate_diversity(self, embeddings):
        """基于嵌入向量的平均值计算多样性"""
        pairwise_dist = [cosine_similarity(e1, e2) 
                        for i, e1 in enumerate(embeddings) 
                        for e2 in embeddings[i+1:]]
        return 1 - np.mean(pairwise_dist)




class GenerationEvaluator:
    def __init__(self, llm):
        self.llm = llm
    
    async def evaluate(self, query, answer, sources):
        """评估生成答案质量"""
        # 1. 事实一致性检查
        faithfulness = await self._check_faithfulness(answer, sources)
        
        # 2. 回答相关性评估
        relevance = await self._check_relevance(query, answer)
        
        # 3. 信息完整性评估
        completeness = await self._check_completeness(answer)
        
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness
        }
    
    async def _check_faithfulness(self, answer, sources):
        """验证答案是否忠实于来源"""
        faithfulness_prompt = f"""

        判断以下答案是否完全基于提供的参考内容（1-5分）：
        参考内容：{" ".join([s.page_content[:500] for s in sources])}
        答案：{answer}
        评分标准：
        - 5分：所有陈述均有明确依据
        - 3分：主要观点有依据，细节可能扩展
        - 1分：存在无依据的重要陈述
        """
        faithfulness_prompt = ChatPromptTemplate.from_messages([
        ("system", faithfulness_prompt),
        ("human", "参考内容：{sources}\n答案：{answer}")
        ])
        faithfulness_chain = faithfulness_prompt | self.llm | StrOutputParser()
        result = await faithfulness_chain.ainvoke({"sources": sources, "answer": answer})
        # 确保result是字符串类型
        if hasattr(result, 'content'):
            result = result.content
        elif not isinstance(result, str):
            result = str(result)
        return result.strip()
    
    async def _check_relevance(self, query, answer):

        """验证答案与问题的相关性"""
        relevance_prompt = f"""
        评估以下回答与问题的相关性（1-5分）：
        问题：{query}
        回答：{answer}
        评分标准：
        - 5分：完全解决所有问题点
        - 3分：解决主要问题但忽略细节
        - 1分：回答与问题无关
        """
        relevance_prompt = ChatPromptTemplate.from_messages([
        ("system", relevance_prompt),
        ("human", "问题：{query}\n回答：{answer}")
        ])
        relanvance_chain = relevance_prompt | self.llm | StrOutputParser()
        result = await relanvance_chain.ainvoke({"query": query, "answer": answer})
        # 确保result是字符串类型
        if hasattr(result, 'content'):
            result = result.content
        elif not isinstance(result, str):
            result = str(result)
        return result.strip()
    
    async def _check_completeness(self, answer):

        """验证答案的完整性"""
        completeness_prompt = f"""
        评估以下回答的完整性（1-5分）： 
        回答：{answer}
        评分标准：
        - 5分：回答完整，包含所有相关信息
        - 3分：回答大部分完整，有部分缺失
        - 1分：回答不完整，缺失重要信息
        """
        completeness_prompt = ChatPromptTemplate.from_messages([
        ("system", completeness_prompt),
        ("human", "回答：{answer}")
        ])
        completeness_chain = completeness_prompt | self.llm | StrOutputParser()
        result = await completeness_chain.ainvoke({"answer": answer})
        # 确保result是字符串类型
        if hasattr(result, 'content'):
            result = result.content
        elif not isinstance(result, str):
            result = str(result)
        return result.strip()

    