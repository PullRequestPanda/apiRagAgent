"""
api_rag手配置文件
Configuration settings for Intelligent Bidding Assistant
"""

import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings(BaseSettings):
    """项目配置类"""
    
    # LLM提供商选择: 'openai' 或 'dashscope'
    llm_provider: Literal['openai', 'dashscope'] = Field(
        default='openai',
        description="LLM提供商选择"
    )
    
    # OpenAI配置
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API密钥"
    )
    openai_base_url: Optional[str] = Field(
        default=None,
        description="OpenAI API基础URL（可选，用于代理）"
    )
    openai_text_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI文本生成模型"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI嵌入模型"
    )
    
    # 阿里云百炼配置
    dashscope_api_key: Optional[str] = Field(
        default="sk-27a3cc36ac9a4684832661c280d65f55",
        description="阿里云百炼API密钥"
    )
    dashscope_text_model: str = Field(
        default="qwen-plus-latest",
        description="通义千问文本生成模型"
    )
    dashscope_embedding_model: str = Field(
        default="text-embedding-v3",
        description="通义千问嵌入模型"
    )
    
    # 向量数据库配置
    vector_store_path: str = Field(
        default="./vector_store",
        description="向量数据库存储路径"
    )
    chunk_size: int = Field(
        default=1000,
        description="文本分块大小"
    )
    chunk_overlap: int = Field(
        default=200,
        description="文本分块重叠大小"
    )
    
    # 检索配置
    retrieval_k: int = Field(
        default=5,
        description="检索返回的文档数量"
    )
    similarity_threshold: float = Field(
        default=0.5,
        description="相似度阈值"
    )

    # 重排序配置
    enable_reranking: bool = Field(
        default=True,
        description="是否启用重排序"
    )
    rerank_model: str = Field(
        default="gte-rerank-v2",
        description="重排序模型名称"
    )
    rerank_top_k: int = Field(
        default=15,
        description="重排序前的候选文档数量"
    )
    rerank_final_k: int = Field(
        default=8,
        description="重排序后的最终文档数量"
    )

    # 增强检索配置
    enable_query_expansion: bool = Field(
        default=False,
        description="是否启用查询扩展"
    )
    enable_multi_round_retrieval: bool = Field(
        default=True,
        description="是否启用多轮检索"
    )
    max_retrieval_rounds: int = Field(
        default=2,
        description="最大检索轮数"
    )

    # 输出配置
    output_dir: str = Field(
        default="./output",
        description="输出文件目录"
    )

    # 向量库隔离配置
    clear_vector_store_on_new_document: bool = Field(
        default=True,
        description="处理新文档时是否清空历史向量数据"
    )

    # JSON解析配置
    enable_json_debug: bool = Field(
        default=True,
        description="是否启用JSON解析调试信息"
    )
    enable_json_fallback: bool = Field(
        default=True,
        description="是否启用JSON解析失败时的备用方法"
    )

    # 日志配置
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    log_file: str = Field(
        default="./logs/bidding_assistant.log",
        description="日志文件路径"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 从环境变量获取API密钥
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.dashscope_api_key:
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

# 全局配置实例
settings = Settings()
