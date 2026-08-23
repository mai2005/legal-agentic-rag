from typing import Any
from src.rag.models import RetrievedChunk

class VectorRetrievalNode:
    def __init__(self, hybrid_retriever, top_k: int=20):
        self.hybrid_retriever = hybrid_retriever
        self.top_k = top_k

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        if not self.hybrid_retriever:
            return {"vector_results": []}
        for item in state.get("sub_queries", []):
            route = state.get("retrieval_routes", {}).get(item["id"])
            if route not in ("vector", "hybrid"):
                continue
            query = item.get("rewritten_query", item["query"])
            retrieved = self.hybrid_retriever.retrieve(query=query, top_k=self.top_k)
            for chunk in retrieved:
                if isinstance(chunk, dict):
                    results.append(chunk)
                else:
                    results.append(self._chunk_to_dict(chunk))
        return {"vector_results": results}

    @staticmethod
    def _chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.document_id,
            "document_id": chunk.document_id,
            "document_title": chunk.document_title,
            "document_type": chunk.document_type,
            "article_id": chunk.article_id,
            "article_number": chunk.article_number,
            "article_title": chunk.article_title,
            "part_number": chunk.part_number,
            "part_title": chunk.part_title,
            "chapter_number": chunk.chapter_number,
            "chapter_title": chunk.chapter_title,
            "content": chunk.content,
            "score": float(chunk.score),
            "source": chunk.source or "vector",
            "metadata": {
                "source_split": chunk.source_split,
            }
        }