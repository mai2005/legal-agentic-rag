import time
from typing import Any
import httpx
from dataclasses import dataclass
from qdrant_client.models import SparseVector
from fastembed import SparseTextEmbedding

@dataclass(slots=True)
class EmbeddingResult:
    dense: list[float]
    sparse: SparseVector

class EmbeddingClientError(RuntimeError):
    pass

class BGEEmbeddingClient:
    def __init__(self, base_url: str, sparse_base_url: str=None, batch_size: int = 32, timeout: float = 60.0, max_retries: int = 3, normalize: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.sparse_base_url = sparse_base_url.rstrip("/") if sparse_base_url else None
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.normalize = normalize

        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(timeout))
        self._sparse_client = httpx.Client(base_url=self.sparse_base_url, timeout=httpx.Timeout(timeout)) if self.sparse_base_url else None
        self.bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BGEEmbeddingClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def embed_document(self, texts: list[str]) -> list[EmbeddingResult]:
        return self._embed(texts)

    def embed_query(self, query: str) -> EmbeddingResult:
        vectors = self._embed([query])
        return vectors[0]

    def _embed(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []

        cleaned_texts  = [self._validate_text(text) for text in texts]
        all_results: list[EmbeddingResult] = []

        for start in range(0, len(cleaned_texts), self.batch_size):
            batch = cleaned_texts[start:start + self.batch_size]
            dense_vectors = self._request_dense(batch)
            sparse_vectors = self._request_sparse(batch)

            if len(dense_vectors) != len(batch) or len(sparse_vectors) != len(batch):
                raise EmbeddingClientError(f"So embedding tra ve khong khop so input")
            
            for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
                all_results.append(EmbeddingResult(dense=dense, sparse=self._convert_sparse_vector(sparse)))

        return all_results

    def _request_dense(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries+1):
            try:
                response = self._client.post(
                    "/embed",
                    json={"inputs": texts, "normalize": self.normalize, "truncate": True}
                )
                response.raise_for_status()

                data: Any = response.json()
                vectors = self._extract_dense(data)
                self._validate_dense(vectors)

                return vectors
            except(httpx.HTTPError, ValueError, TypeError, EmbeddingClientError) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(2**(attempt-1))
        raise EmbeddingClientError(f"Khong the lay embedding sau {self.max_retries} lan thu") from last_error
    
    # def _request_sparse(self, texts: list[str]) -> list[list[dict[str, Any]]]:
    #     last_error: Exception | None = None
    #     for attempt in range(1, self.max_retries+1):
    #         try:
    #             response = self._sparse_client.post(
    #                 "/embed_sparse",
    #                 json={"inputs": texts}
    #             )
    #             response.raise_for_status()

    #             data: Any = response.json()
    #             self._validate_sparse(data)

    #             return data
    #         except(httpx.HTTPError, ValueError, TypeError, EmbeddingClientError) as error:
    #             last_error = error
    #             if attempt < self.max_retries:
    #                 time.sleep(2**(attempt-1))
    #     raise EmbeddingClientError(f"Khong the lay embedding sau {self.max_retries} lan thu") from last_error

    def _request_sparse(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        embeddings = list(self.bm25_model.embed(texts))
        
        formatted_sparse_vectors = []
        for emb in embeddings:
            vector_data = [
                {"index": int(idx), "value": float(val)}
                for idx, val in zip(emb.indices, emb.values)
            ]
            formatted_sparse_vectors.append(vector_data)
        return formatted_sparse_vectors

    @staticmethod
    def _convert_sparse_vector(raw: list[dict[str, Any]]) -> SparseVector:
        indices = []
        values = []

        for item in raw:
            if "index" not in item or "value" not in item:
                raise EmbeddingClientError("Dinh dang sparse embedding khong hop le")
            indices.append(int(item["index"]))
            values.append(float(item["value"]))
        
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def _extract_dense(data: Any) -> list[list[float]]:
        if isinstance(data, list):
            return data

        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]

        if isinstance(data, dict) and "data" in data:
            return [item["embedding"] for item in data["data"]]

        raise EmbeddingClientError(f"Khong nhan dien duoc response embedding: {type(data)}")

    @staticmethod
    def _validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Input embedding phai la str, nhan duoc {type(text)}")

        text = text.strip()

        if not text:
            raise ValueError("Khong duoc embed chuoi rong")

        return text

    @staticmethod
    def _validate_dense(vectors: list[list[float]]) -> None:
        if not vectors:
            raise EmbeddingClientError("Embedding server tra ve danh sach rong")

        dimensions = {len(vector) for vector in vectors}

        if len(dimensions) != 1:
            raise EmbeddingClientError(f"Cac embedding khong cung dimension: {dimensions}")

        if 0 in dimensions:
            raise EmbeddingClientError("Embedding co dimensions = 0")

    @staticmethod
    def _validate_sparse(vectors: Any) -> None:
        if not isinstance(vectors, list):
            raise EmbeddingClientError(f"Sparse response phai tra ve list")

        for vector in vectors:
            if not isinstance(vector, list):
                raise EmbeddingClientError("Sparse vector phai tra ve list")

            for token in vector:
                if not isinstance(token, dict) or "index" not in token or "value" not in token:
                    raise EmbeddingClientError("Phan tu cua sparse vector khong hop le")