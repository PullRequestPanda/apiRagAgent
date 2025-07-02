"""
LLM工厂类
Factory for creating LLM and embedding models
"""

from typing import Optional, Any
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from loguru import logger
import dashscope
from config.settings import settings
from pydantic import SecretStr

class LLMFactory:
    """LLM和嵌入模型工厂类"""

    @staticmethod
    def create_llm(provider: Optional[str] = None) -> Any:
        """
        创建LLM实例
        Args:
            provider: LLM提供商 ('openai' 或 'dashscope')
        Returns:
            LLM实例
        """
        if provider is None:
            provider = settings.llm_provider
        try:
            if provider == 'openai':
                return LLMFactory._create_openai_llm()
            elif provider == 'dashscope':
                return LLMFactory._create_dashscope_llm()
            else:
                raise ValueError(f"不支持的LLM提供商: {provider}")
        except Exception as e:
            logger.error(f"创建LLM失败: {e}")
            raise

    @staticmethod
    def create_embeddings(provider: Optional[str] = None) -> Embeddings:
        """
        创建嵌入模型实例
        Args:
            provider: 嵌入模型提供商 ('openai' 或 'dashscope')
        Returns:
            Embeddings实例
        """
        if provider is None:
            provider = settings.llm_provider
        try:
            if provider == 'openai':
                return LLMFactory._create_openai_embeddings()
            elif provider == 'dashscope':
                return LLMFactory._create_dashscope_embeddings()
            else:
                raise ValueError(f"不支持的嵌入模型提供商: {provider}")
        except Exception as e:
            logger.error(f"创建嵌入模型失败: {e}")
            raise

    @staticmethod
    def _create_openai_llm() -> ChatOpenAI:
        """创建OpenAI LLM"""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API密钥未设置")
        kwargs = {
            "model": settings.openai_text_model,
            "api_key": settings.openai_api_key,
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        logger.info(f"创建OpenAI LLM: {settings.openai_text_model}")
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _create_openai_embeddings() -> OpenAIEmbeddings:
        """创建OpenAI嵌入模型"""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API密钥未设置")
        api_key = SecretStr(settings.openai_api_key)
        if settings.openai_base_url:
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=api_key,
                base_url=settings.openai_base_url
            )
        else:
            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=api_key
            )

    @staticmethod
    def _create_dashscope_llm() -> Tongyi:
        """创建阿里云通义千问LLM"""
        if not settings.dashscope_api_key:
            raise ValueError("阿里云百炼API密钥未设置")
        logger.info(f"创建阿里云通义千问LLM: {settings.dashscope_text_model}")
        return Tongyi(
            model_name=settings.dashscope_text_model,
            dashscope_api_key=settings.dashscope_api_key,
            temperature=0.1,
            max_tokens=4000,
        )

    @staticmethod
    def _create_dashscope_embeddings() -> DashScopeEmbeddings:
        """创建阿里云通义嵌入模型"""
        if not settings.dashscope_api_key:
            raise ValueError("阿里云百炼API密钥未设置")
        logger.info(f"创建阿里云通义嵌入模型: {settings.dashscope_embedding_model}")
        return DashScopeEmbeddings(
            model=settings.dashscope_embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )

    @staticmethod
    def test_connection(provider: Optional[str] = None) -> bool:
        """
        测试LLM连接
        Args:
            provider: LLM提供商
        Returns:
            bool: 连接是否成功
        """
        try:
            llm = LLMFactory.create_llm(provider)
            # 简单的测试调用
            if hasattr(llm, 'invoke'):
                response = llm.invoke("测试连接")
            else:
                response = llm("测试连接")
            logger.info(f"LLM连接测试成功: {provider}")
            return True
        except Exception as e:
            logger.error(f"LLM连接测试失败: {provider}, 错误: {e}")
            return False

    @staticmethod
    def test_embeddings(provider: Optional[str] = None) -> bool:
        """
        测试嵌入模型连接
        Args:
            provider: 嵌入模型提供商
        Returns:
            bool: 连接是否成功
        """
        try:
            embeddings = LLMFactory.create_embeddings(provider)
            # 简单的测试调用
            test_text = ["测试嵌入模型连接"]
            result = embeddings.embed_documents(test_text)
            if result and len(result) > 0:
                logger.info(f"嵌入模型连接测试成功: {provider}")
                return True
            else:
                logger.error(f"嵌入模型返回空结果: {provider}")
                return False
        except Exception as e:
            logger.error(f"嵌入模型连接测试失败: {provider}, 错误: {e}")
            return False

    @staticmethod
    def test_embeddings_query(provider: Optional[str] = None) -> bool:
        """
        测试嵌入模型连接（单条query）
        Args:
            provider: 嵌入模型提供商
        Returns:
            bool: 连接是否成功
        """
        try:
            embeddings = LLMFactory.create_embeddings(provider)
            test_embedding = embeddings.embed_query("测试连接")
            logger.info(f"嵌入模型连接测试成功: {provider}")
            return True
        except Exception as e:
            logger.error(f"嵌入模型连接测试失败: {provider}, 错误: {e}")
            return False