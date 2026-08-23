from typing import Any
from qdrant_client.models import FieldCondition, Filter, MatchValue
from src.rag.models import RetrievedChunk
from src.rag.vector_retriever import VectorRetriever
from src.rag.fusion import RRF
from src.rag.reranker import BGERerankerClient

class HybridRetriever:
    def __init__(self, vector_retriever: VectorRetriever, rrf: RRF | None=None, reranker: BGERerankerClient | None=None) -> None:
        self.vector_retriever = vector_retriever
        self.rrf = rrf or RRF()
        self.reranker = reranker

    def retrieve(self, query: str, metadata: dict[str, Any] | None=None, top_k: int=15, top_k_dense: int=50, top_k_sparse: int=50, score_threshold: float | None=None, top_k_rerank: int=45) -> list[RetrievedChunk]:
        embedding = self.vector_retriever.embed_query(query)
        query_filter = self._build_filter(metadata)

        dense_results = self.vector_retriever.dense_search(embedding, query_filter, top_k=top_k_dense, score_threshold=score_threshold)
        sparse_results = self.vector_retriever.sparse_search(embedding, query_filter, top_k=top_k_sparse, score_threshold=score_threshold)

        fuse_limit = top_k_rerank if self.reranker else top_k
        fused = self.rrf.fuse(rankings=[dense_results, sparse_results], top_k=fuse_limit)

        if self.reranker:
            fused = self.reranker.rerank(query=query, chunks=fused, top_k=top_k)
        return fused

    @staticmethod
    def _build_filter(metadata: dict[str, Any] | None) -> Filter | None:
        if not metadata:
            return None
        
        conditions: list[FieldCondition] = []

        for key, value in metadata.items():
            if value is None:
                continue
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        
        if not conditions:
            return None

        return Filter(must=conditions)