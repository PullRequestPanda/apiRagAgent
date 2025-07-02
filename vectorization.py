from langchain_core.documents import Document
import json
# from qwen_embeddings import QwenSentenceTransformerEmbeddings
from src.vector_stores import VectorStoreManager
from src.llm_factory import LLMFactory

with open("data/api.json", "r", encoding="utf-8") as f:
    api_docs = json.load(f)

docs = []
for api in api_docs:
    # 构建用于 embedding 的
    # 文本内容（自然语言检索）
    param_str = "; ".join(f"{p['name']}({p['type']}, 必填: {p['required']}) - {p['description']}" for p in api["params"])
    content = f"{api['name']} | {api['description']} | 方法: {api['method']} {api['endpoint']} | 参数: {param_str}"

    # 构建 metadata，结构字段转为 JSON 字符串
    metadata = {
        "name": api["name"],
        "description": api["description"],
        "method": api["method"],
        "endpoint": api["endpoint"],
        "params_json": json.dumps(api["params"], ensure_ascii=False)  # 👈 转为 JSON 字符串
    }

    docs.append(Document(page_content=content, metadata=metadata))
my_embeddings = LLMFactory.create_embeddings(provider='dashscope')

# 然后你可以继续使用：
vsm = VectorStoreManager(embeddings=my_embeddings)
vsm.create_vector_store(docs, collection_name='api_docs')



