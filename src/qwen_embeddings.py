from langchain.embeddings.base import Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer

class QwenSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self.model.encode(texts, batch_size=32, convert_to_numpy=True)]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
