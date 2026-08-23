from src.rag.models import RetrievedChunk
from collections import defaultdict

class RRF:
    def __init__(self, k: int=60) -> None:
        self.k = k

    def fuse(self, rankings: list[list[RetrievedChunk]], top_k: int | None=None) -> list[RetrievedChunk]:
        if not rankings:
            return []

        scores: dict[str, float] = defaultdict(float)
        chunk_lookup: dict[str, RetrievedChunk] = {}

        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                chunk_lookup[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] += 1.0/(self.k+rank)
        
        fused = sorted(chunk_lookup.values(), key=lambda chunk: scores[chunk.chunk_id], reverse=True)

        if top_k is not None:
            fused = fused[:top_k]
        
        return fused