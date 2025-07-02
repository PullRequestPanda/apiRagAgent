from src.vector_stores import VectorStoreManager
#本地 qwen_embeddings
# from qwen_embeddings import QwenSentenceTransformerEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ConversationBufferMemory
import json
import os
import re
from typing import Optional
from pydantic import SecretStr
from src.llm_factory import LLMFactory
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

class ApiRagAgent:
    def __init__(self):
        # 初始化向量存储与重排序器
        self.embeddings = LLMFactory.create_embeddings(provider='dashscope')
        self.vsm = VectorStoreManager(embeddings=self.embeddings)
        self.vsm.load_vector_store(collection_name="api_docs")

        # 初始化智谱千问（GLM-4）模型
        self.llm = LLMFactory.create_llm(provider='dashscope')
        # 初始化记忆（存储对话历史）
        self.memory = ConversationBufferMemory()

        # 提示词模板 - 要求模型只返回纯 JSON
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """你是一个智能接口助手，目标是根据用户需求和接口文档，直接返回一个 JSON 请求体，符合后端接口调用规范。

        接口信息如下：
        名称: {name}
        描述: {description}
        方法: {method}
        地址: {endpoint}
        参数说明: {params_json}

        请根据用户输入构造调用请求体，格式如下：
        {{ 
        "method": "GET" | "POST" | "DELETE" | "PUT",
        "path": "/api/xxx/{{{{id}}}} 或 /api/xxx",
        "body": {{ ... }}  // 可选，仅POST/PUT才需要
        }}

        请遵循以下要求：
        1. 直接返回 JSON 对象，不需要解释说明
        2. 仅输出 JSON，不需要使用 markdown 代码块
        3. 若用户参数不全，仅输出已有字段和一个 "missing" 字段，列出缺失字段名（字符串数组）

        示例：

        用户输入：查询用户id为u_456的用户信息
        返回：
        {{"method": "GET", "path": "/api/v1/users/u_456"}}

        用户输入：帮我给用户123创建一个订单，买3个商品编号为p_888的商品，备注写加急处理
        返回：
        {{"method": "POST", "path": "/api/v1/orders", "body": {{"user_id": "123", "product_id": "p_888", "quantity": 3, "note": "加急处理"}}}}

        用户输入：把用户编号是u_999的账户删掉
        返回：
        {{"method": "DELETE", "path": "/api/v1/users/u_999"}}
        ⚠️ 注意：不要使用 markdown 语法包裹（如 ```json），否则将无法识别并报错！
        """
            ),
            HumanMessagePromptTemplate.from_template("用户需求: {user_query}")
        ])


        # 构造无 memory 的 chain
        self.api_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True
        )

    def retrieve_apis(self, query: str, top_k=5):
        """执行 API 检索"""
        return self.vsm.enhanced_similarity_search(query, k=top_k, use_reranking=True)

    def query_to_api_json(self, query: str, override_params: Optional[dict] = None) -> dict:
        """将自然语言用户请求转换为符合规范的 JSON 请求体"""
        results = self.retrieve_apis(query)
        if not results:
            return {"status": "error", "message": "未找到相关接口"}

        top_doc = results[0][0]
        metadata = top_doc.metadata
        params = json.loads(metadata["params_json"])

        # 构建 LLMChain 输入
        inputs = {
            "name": metadata["name"],
            "description": metadata["description"],
            "method": metadata["method"],
            "endpoint": metadata["endpoint"],
            "params_json": json.dumps(params, ensure_ascii=False),
            "user_query": query,
        }

        # 调用大模型
        response = self.api_chain.invoke(inputs)
        print("🧪 Raw response from chain:", response)

        raw_text = response.get("text", "").strip()

        # 清理 markdown 语法或意外包裹
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        raw_text = raw_text.strip()

        try:
            parsed = json.loads(raw_text)
            # 检查必须字段
            missing_fields = []
            for field in ["method", "path"]:
                if field not in parsed:
                    missing_fields.append(field)
            if missing_fields:
                parsed["missing"] = missing_fields
            return parsed
        except Exception as e:
            return {
                "status": "error",
                "message": f"解析模型响应失败: {str(e)}",
                "raw": raw_text
            }

app = FastAPI()

# 实例化一次，避免每次请求都加载模型
agent = ApiRagAgent()

@app.post("/api/query")
async def query_api(request: Request):
    data = await request.json()
    query = data.get("query", "")
    if not query:
        return JSONResponse({"status": "error", "message": "缺少 query 参数"}, status_code=400)
    result = agent.query_to_api_json(query)
    return JSONResponse(result)

if __name__ == "__main__":
    # 启动FastAPI服务
    uvicorn.run("api_rag_agent:app", host="0.0.0.0", port=8000, reload=True)