from typing import Any
from src.rag.models import RetrievedChunk

class RerankerNode:
    def __init__(self, reranker=None, top_k: int=8):
        self.reranker = reranker
        self.top_k = top_k

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        fused = state.get("fused_results", [])
        if not fused:
            return {"reranked_results": []}

        if not self.reranker:
            return {"reranked_results": fused[:self.top_k]}

        chunks: list[RetrievedChunk] = []
        for item in fused:
            if isinstance(item, RetrievedChunk):
                chunks.append(item)
            else:
                chunks.append(RetrievedChunk(
                    chunk_id=str(item.get("chunk_id", "")),
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                    document_id=str(item.get("doc_id") or item.get("document_id", "")),
                    document_title=item.get("document_title"),
                    document_type=item.get("document_type"),
                    article_id=item.get("article_id"),
                    article_number=item.get("article_number"),
                    article_title=item.get("article_title"),
                    part_number=item.get("part_number"),
                    part_title=item.get("part_title"),
                    chapter_number=item.get("chapter_number"),
                    chapter_title=item.get("chapter_title"),
                    source=item.get("source"),
                    source_split=item.get("metadata", {}).get("source_split")
                ))

        reranked_chunks = self.reranker.rerank(query=query, chunks=chunks, top_k=self.top_k)

        reranked_results: list[dict[str, Any]] = []
        for chunk in reranked_chunks:
            reranked_results.append({
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
                "source": chunk.source or "reranker",
                "metadata": {
                    "source_split": chunk.source_split,
                }
            })

        return {"reranked_results": reranked_results}