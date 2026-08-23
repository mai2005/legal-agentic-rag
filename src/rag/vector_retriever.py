from typing import Any
from src.storage.qdrant_store import QdrantVectorStore
from src.embedding.embedding_client import BGEEmbeddingClient, EmbeddingResult
from src.rag.models import RetrievedChunk

class VectorRetriever:
    def __init__(self, vector_store: QdrantVectorStore, embedding_client: BGEEmbeddingClient) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def embed_query(self, query: str) -> EmbeddingResult:
        query = query.strip()
        if not query:
            raise ValueError("Query khong duoc rong")
        return self.embedding_client.embed_query(query)

    def dense_search(self, embedding: EmbeddingResult, query_filter, top_k: int=20, score_threshold: float | None=None) -> list[RetrievedChunk]:
        points = self.vector_store.search_dense(query_vector=embedding.dense, top_k=top_k, query_filter=query_filter, score_threshold=score_threshold)
        return self._convert_points(points)

    def sparse_search(self, embedding: EmbeddingResult, query_filter, top_k: int=20, score_threshold: float | None=None) -> list[RetrievedChunk]:
        points = self.vector_store.search_sparse(query_vector=embedding.sparse, top_k=top_k, query_filter=query_filter, score_threshold=score_threshold)
        return self._convert_points(points)

    def _convert_points(self, points: list[Any]) -> list[RetrievedChunk]:
        return [self._to_retrieved_chunk(point.payload or {}, float(point.score)) for point in points]

    @staticmethod
    def _to_retrieved_chunk(payload: dict[str, Any], score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id = str(payload.get("chunk_id", "")),
            content = str(payload.get("content", "")),
            score = score,
            document_id = str(payload.get("document_id", "")),
            document_title = payload.get("document_title"),
            document_type = payload.get("document_type"),
            article_id = payload.get("article_id"),
            article_number = payload.get("article_number"),
            article_title = payload.get("article_title"),
            part_number = payload.get("part_number"),
            part_title = payload.get("part_title"),
            chapter_number = payload.get("chapter_number"),
            chapter_title = payload.get("chapter_title"),
            source = payload.get("source"),
            source_split = payload.get("source_split")
        )