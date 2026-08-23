from typing import Any
import httpx
import copy
import time
from src.rag.models import RetrievedChunk

class RerankerClientError(RuntimeError):
    """Raise khi reranker service tra ve phan hoi khong hop le."""

class BGERerankerClient:
    def __init__(self, base_url: str, endpoint: str="/rerank", timeout: float=60.0, max_retries: int=3, raw_scores: bool=False) -> None:
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.raw_scores = raw_scores
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BGERerankerClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None=None) -> list[RetrievedChunk]:
        query = self._validate_query(query)
        if not chunks:
            return []
        texts = [str(chunk.content) for chunk in chunks]
        results = self._request_with_retry(query=query, texts=texts)
        reranked_chunks: list[RetrievedChunk] = []
        for result in results:
            index = int(result["index"])
            score = float(result["score"])
            if index<0 or index>=len(chunks):
                raise RerankerClientError(f"Reranker index khogn hop le: {index}")
            chunk = copy.copy(chunks[index])
            chunk.score = score
            reranked_chunks.append(chunk)
        reranked_chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        if top_k is not None:
            reranked_chunks = reranked_chunks[:top_k]
        return reranked_chunks

    def _request_with_retry(self, query: str, texts: list[str]) -> list[dict[str, Any]]:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries+1):
            try:
                response = self._client.post(self.endpoint,
                json={
                    "query" : query,
                    "texts": texts,
                    "raw_scores": self.raw_scores
                })
                response.raise_for_status()
                data = response.json()
                results = self._extract_results(data)
                self._validate_results(results)
                return results
            except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError, RerankerClientError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(2**(attempt-1))
        raise RerankerClientError(f"Fail to rerank sau {self.max_retries} lan thu.") from last_error

    @staticmethod
    def _extract_results(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data        
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        raise RerankerClientError(f"Reranker tra ve dinh dang khong duoc ho tro: {type(data)}.")

    @staticmethod
    def _validate_query(query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query khong the rong.")
        return query

    @staticmethod
    def _validate_results(results: list[dict[str, Any]]) -> None:
        if not results:
            raise RerankerClientError("Reranker tra ve rong.")

        for result in results:
            if not isinstance(result, dict):
                raise RerankerClientError("Reranker phai tra ve dictionary.")
            if "index" not in result:
                raise RerankerClientError("Reranker tra ve thieu key 'index'.")
            if "score" not in result:
                raise RerankerClientError("Reranker tra ve thieu key 'score'.")
