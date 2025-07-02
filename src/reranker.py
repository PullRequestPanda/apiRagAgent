"""
重排序模型工具
Reranking utilities for improving RAG retrieval quality
"""

from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from loguru import logger
import dashscope
from dashscope import TextReRank
from config.settings import settings
import time


class RerankerManager:
    """重排序管理器"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化重排序管理器
        
        Args:
            model_name: 重排序模型名称
        """
        self.model_name = model_name or settings.rerank_model
        self.api_key = settings.dashscope_api_key
        
        if not self.api_key:
            logger.warning("阿里云百炼API密钥未设置，重排序功能将被禁用")
            self.enabled = False
        else:
            dashscope.api_key = self.api_key
            self.enabled = True
            logger.info(f"重排序管理器初始化成功，模型: {self.model_name}")
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 重排序分数)列表
        """
        if not self.enabled:
            logger.warning("重排序功能未启用，返回原始文档")
            return [(doc, 1.0) for doc in documents]
        
        if not documents:
            return []
        
        if top_k is None:
            top_k = settings.rerank_final_k
        
        try:
            # 准备文档文本，过滤空文档
            doc_texts = []
            valid_documents = []

            for doc in documents:
                content = doc.page_content.strip()
                if content and len(content) > 10:  # 过滤太短的文档
                    doc_texts.append(content)
                    valid_documents.append(doc)

            if not doc_texts:
                logger.warning("没有有效的文档用于重排序")
                return [(doc, 1.0) for doc in documents[:top_k]]

            logger.debug(f"开始重排序，查询: {query[:50]}..., 有效文档数量: {len(doc_texts)}")

            # 调用阿里云百炼重排序API
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(doc_texts)),
                return_documents=True
            )
            
            if response.status_code != 200:
                logger.error(f"重排序API调用失败: {response.message}")
                return [(doc, 1.0) for doc in documents[:top_k]]
            
            # 解析重排序结果
            reranked_results = []
            for result in response.output.results:
                doc_index = result.index
                score = result.relevance_score

                if 0 <= doc_index < len(valid_documents):
                    reranked_results.append((valid_documents[doc_index], score))
            
            logger.info(f"重排序完成，返回 {len(reranked_results)} 个文档")
            return reranked_results
            
        except Exception as e:
            logger.error(f"重排序过程中发生错误: {e}")
            # 发生错误时返回原始文档
            return [(doc, 1.0) for doc in documents[:top_k]]
    
    def rerank_with_scores(
        self, 
        query: str, 
        doc_score_pairs: List[Tuple[Document, float]], 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        对带分数的文档进行重排序
        
        Args:
            query: 查询文本
            doc_score_pairs: (文档, 原始分数)列表
            top_k: 返回的文档数量
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 原始分数, 重排序分数)列表
        """
        if not self.enabled:
            logger.warning("重排序功能未启用，返回原始文档")
            return [(doc, orig_score, orig_score) for doc, orig_score in doc_score_pairs]
        
        if not doc_score_pairs:
            return []
        
        # 提取文档和原始分数
        documents = [doc for doc, _ in doc_score_pairs]
        original_scores = [score for _, score in doc_score_pairs]
        
        # 进行重排序
        reranked_results = self.rerank_documents(query, documents, top_k)
        
        # 合并原始分数和重排序分数
        final_results = []
        for doc, rerank_score in reranked_results:
            # 找到对应的原始分数
            orig_score = 1.0
            for orig_doc, orig_score_val in doc_score_pairs:
                if orig_doc.page_content == doc.page_content:
                    orig_score = orig_score_val
                    break
            
            final_results.append((doc, orig_score, rerank_score))
        
        return final_results
    
    def batch_rerank(
        self, 
        queries_docs: List[Tuple[str, List[Document]]], 
        top_k: Optional[int] = None
    ) -> List[List[Tuple[Document, float]]]:
        """
        批量重排序
        
        Args:
            queries_docs: (查询, 文档列表)的列表
            top_k: 每个查询返回的文档数量
            
        Returns:
            List[List[Tuple[Document, float]]]: 每个查询的重排序结果列表
        """
        results = []
        for query, documents in queries_docs:
            reranked = self.rerank_documents(query, documents, top_k)
            results.append(reranked)
            
            # 添加小延迟避免API限流
            time.sleep(0.1)
        
        return results


class HybridRetriever:
    """混合检索器，结合向量检索和重排序"""
    
    def __init__(self, vector_store, reranker: Optional[RerankerManager] = None):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储
            reranker: 重排序管理器
        """
        self.vector_store = vector_store
        self.reranker = reranker or RerankerManager()
        
    def retrieve_and_rerank(
        self, 
        query: str, 
        initial_k: Optional[int] = None, 
        final_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        检索并重排序
        
        Args:
            query: 查询文本
            initial_k: 初始检索的文档数量
            final_k: 最终返回的文档数量
            
        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        if initial_k is None:
            initial_k = settings.rerank_top_k
        if final_k is None:
            final_k = settings.rerank_final_k
        
        try:
            # 第一步：向量检索
            logger.debug(f"开始向量检索，查询: {query[:50]}..., k={initial_k}")
            vector_results = self.vector_store.similarity_search_with_score(query, k=initial_k)
            
            if not vector_results:
                logger.warning("向量检索未返回任何结果")
                return []
            
            # 第二步：重排序
            if settings.enable_reranking and self.reranker.enabled:
                logger.debug(f"开始重排序，候选文档数量: {len(vector_results)}")
                reranked_results = self.reranker.rerank_with_scores(
                    query, vector_results, final_k
                )
                return reranked_results
            else:
                # 如果重排序未启用，直接返回向量检索结果
                return [(doc, score, score) for doc, score in vector_results[:final_k]]
                
        except Exception as e:
            logger.error(f"混合检索过程中发生错误: {e}")
            return []
    
    def multi_query_retrieve(
        self, 
        queries: List[str], 
        final_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        多查询检索并去重
        
        Args:
            queries: 查询列表
            final_k: 最终返回的文档数量
            
        Returns:
            List[Tuple[Document, float, float]]: 去重后的文档列表
        """
        if final_k is None:
            final_k = settings.rerank_final_k
        
        all_results = []
        seen_contents = set()
        
        for query in queries:
            results = self.retrieve_and_rerank(query, final_k=final_k * 2)  # 获取更多候选
            
            for doc, vec_score, rerank_score in results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append((doc, vec_score, rerank_score))
        
        # 按重排序分数排序并返回top_k
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:final_k]
