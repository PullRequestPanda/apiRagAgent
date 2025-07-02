"""
向量存储工具
Vector store utilities for RAG system
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from loguru import logger
import os
import shutil
from config.settings import settings

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, embeddings: Embeddings, persist_directory: Optional[str] = None):
        """
        初始化向量存储管理器
        
        Args:
            embeddings: 嵌入模型
            persist_directory: 持久化目录
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory or settings.vector_store_path
        self.vector_store: Optional[Chroma] = None
        
        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document], collection_name: str = "bidding_docs") -> Chroma:
        """
        创建向量存储
        
        Args:
            documents: 文档列表
            collection_name: 集合名称
            
        Returns:
            Chroma: 向量存储实例
        """
        try:
            logger.info(f"开始创建向量存储，文档数量: {len(documents)}")
            
            # 创建向量存储
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            logger.info(f"向量存储创建成功，集合名称: {collection_name}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {e}")
            raise
    
    def load_vector_store(self, collection_name: str = "bidding_docs") -> Optional[Chroma]:
        """
        加载现有的向量存储
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Optional[Chroma]: 向量存储实例
        """
        try:
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                logger.info(f"向量存储加载成功，集合名称: {collection_name}")
                return self.vector_store
            else:
                logger.warning(f"向量存储目录不存在: {self.persist_directory}")
                return None
                
        except Exception as e:
            logger.error(f"加载向量存储失败: {e}")
            return None
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        向现有向量存储添加文档
        
        Args:
            documents: 要添加的文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"成功添加{len(documents)}个文档到向量存储")
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            raise
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            List[Document]: 相似文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        if k is None:
            k = settings.retrieval_k
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成，查询: {query[:50]}..., 返回{len(results)}个结果")
            return results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            raise
    
    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回文档数量
            
        Returns:
            List[Tuple[Document, float]]: (文档, 相似度分数)列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        if k is None:
            k = settings.retrieval_k
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # 过滤低相似度结果
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= settings.similarity_threshold
            ]
            
            logger.debug(f"带分数相似度搜索完成，查询: {query[:50]}..., 返回{len(filtered_results)}个结果")
            return filtered_results
            
        except Exception as e:
            logger.error(f"带分数相似度搜索失败: {e}")
            raise
    
    def create_retriever(self, search_type: str = "similarity", search_kwargs: Optional[dict] = None):
        """
        创建检索器
        
        Args:
            search_type: 搜索类型
            search_kwargs: 搜索参数
            
        Returns:
            检索器实例
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        if search_kwargs is None:
            search_kwargs = {"k": settings.retrieval_k}
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )
            logger.info(f"检索器创建成功，搜索类型: {search_type}")
            return retriever
            
        except Exception as e:
            logger.error(f"创建检索器失败: {e}")
            raise
    
    def clear_vector_store(self) -> None:
        """清空向量存储"""
        try:
            # 先关闭现有的向量存储连接
            if self.vector_store is not None:
                try:
                    # 尝试删除集合
                    self.vector_store.delete_collection()
                except:
                    pass
                self.vector_store = None

            # 等待一下让文件句柄释放
            import time
            time.sleep(0.5)

            if os.path.exists(self.persist_directory):
                # 简化清理逻辑，直接清空内容而不是删除整个目录
                self._clear_directory_contents(self.persist_directory)
                logger.info("向量存储已清空")
            else:
                # 如果目录不存在，直接创建
                os.makedirs(self.persist_directory, exist_ok=True)
                logger.info("创建新的向量存储目录")

        except Exception as e:
            logger.error(f"清空向量存储失败: {e}")
            # 如果清空失败，尝试重置到默认目录
            try:
                default_dir = "./vector_store"
                if self.persist_directory != default_dir:
                    self.persist_directory = default_dir
                    os.makedirs(self.persist_directory, exist_ok=True)
                    logger.info(f"重置到默认向量存储目录: {self.persist_directory}")
                else:
                    # 如果已经是默认目录，则不创建新目录，直接使用现有的
                    logger.warning("使用现有向量存储目录，可能存在历史数据")
            except Exception as e2:
                logger.error(f"重置向量存储目录也失败: {e2}")
                raise e

    def _clear_directory_contents(self, directory: str) -> None:
        """清空目录内容而不删除目录本身"""
        import time
        max_retries = 3

        for attempt in range(max_retries):
            try:
                for item in os.listdir(directory):
                    item_path = os.path.join(directory, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                return
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"清空目录内容失败，重试 {attempt + 1}/{max_retries}: {e}")
                    time.sleep(1)
                else:
                    # 最后一次尝试失败，抛出异常让上层处理
                    raise e

    def clear_collection(self, collection_name: str) -> None:
        """
        清空指定集合

        Args:
            collection_name: 要清空的集合名称
        """
        try:
            if os.path.exists(self.persist_directory):
                # 尝试删除指定集合
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                vector_store.delete_collection()
                logger.info(f"集合 {collection_name} 已清空")

        except Exception as e:
            logger.warning(f"清空集合 {collection_name} 失败: {e}")
            # 如果单独清空集合失败，可以选择清空整个向量存储
            logger.info("尝试清空整个向量存储...")
            self.clear_vector_store()

    def create_isolated_vector_store(self, documents: List[Document], document_path: str) -> Chroma:
        """
        为单个文档创建完全隔离的向量存储

        Args:
            documents: 文档列表
            document_path: 文档路径，用于生成唯一标识

        Returns:
            Chroma: 向量存储实例
        """
        try:
            # 清空现有的向量存储，确保完全隔离
            logger.info("清空历史向量数据，确保新文档处理的独立性...")
            self.clear_vector_store()

            # 基于文档路径和时间戳生成唯一集合名
            import time
            import uuid
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:8]
            file_hash = abs(hash(document_path)) % 10000
            collection_name = f"isolated_doc_{file_hash}_{timestamp}_{unique_id}"

            logger.info(f"为文档创建隔离的向量存储: {document_path}")
            logger.info(f"集合名称: {collection_name}")

            # 等待一下确保目录清理完成
            time.sleep(0.5)

            # 创建新的向量存储
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )

            logger.info(f"隔离向量存储创建成功，文档数量: {len(documents)}")
            return self.vector_store

        except Exception as e:
            logger.error(f"创建隔离向量存储失败: {e}")
            # 如果隔离模式失败，回退到传统模式
            logger.warning("隔离模式失败，回退到传统模式...")
            try:
                fallback_collection = f"fallback_{abs(hash(document_path)) % 10000}"
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=fallback_collection
                )
                logger.info(f"回退模式创建成功，集合名称: {fallback_collection}")
                return self.vector_store
            except Exception as e2:
                logger.error(f"回退模式也失败: {e2}")
                raise e
    
    def get_collection_info(self) -> dict:
        """
        获取集合信息

        Returns:
            dict: 集合信息
        """
        if self.vector_store is None:
            return {"status": "未初始化"}

        try:
            collection = self.vector_store._collection
            return {
                "name": collection.name,
                "count": collection.count(),
                "status": "已初始化"
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"status": "错误", "error": str(e)}

    def enhanced_similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        增强的相似度搜索，支持重排序

        Args:
            query: 查询文本
            k: 返回文档数量
            use_reranking: 是否使用重排序

        Returns:
            List[Tuple[Document, float, float]]: (文档, 向量分数, 重排序分数)列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")

        if k is None:
            k = settings.rerank_final_k
        if use_reranking is None:
            use_reranking = settings.enable_reranking

        try:
            # 第一步：向量检索
            initial_k = settings.rerank_top_k if use_reranking else k
            vector_results = self.vector_store.similarity_search_with_score(query, k=initial_k)

            if not vector_results:
                return []

            # 第二步：重排序（如果启用）
            if use_reranking and settings.enable_reranking:
                try:
                    from reranker import RerankerManager
                    reranker = RerankerManager()

                    if reranker.enabled:
                        reranked_results = reranker.rerank_with_scores(query, vector_results, k)
                        logger.debug(f"增强检索完成，重排序后返回 {len(reranked_results)} 个文档")
                        return reranked_results
                    else:
                        logger.warning("重排序器未启用，使用向量检索结果")

                except ImportError:
                    logger.warning("重排序模块导入失败，使用向量检索结果")
                except Exception as e:
                    logger.error(f"重排序过程失败: {e}，使用向量检索结果")

            # 返回向量检索结果
            return [(doc, score, score) for doc, score in vector_results[:k]]

        except Exception as e:
            logger.error(f"增强相似度搜索失败: {e}")
            raise

    def multi_query_search(
        self,
        queries: List[str],
        k_per_query: Optional[int] = None,
        final_k: Optional[int] = None
    ) -> List[Tuple[Document, float, float]]:
        """
        多查询搜索并合并结果

        Args:
            queries: 查询列表
            k_per_query: 每个查询返回的文档数量
            final_k: 最终返回的文档数量

        Returns:
            List[Tuple[Document, float, float]]: 合并后的文档列表
        """
        if k_per_query is None:
            k_per_query = settings.retrieval_k
        if final_k is None:
            final_k = settings.rerank_final_k

        all_results = []
        seen_contents = set()

        for query in queries:
            try:
                query_results = self.enhanced_similarity_search(query, k=k_per_query)

                for doc, vec_score, rerank_score in query_results:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        all_results.append((doc, vec_score, rerank_score))

            except Exception as e:
                logger.error(f"查询 '{query}' 搜索失败: {e}")
                continue

        # 按重排序分数排序
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:final_k]
