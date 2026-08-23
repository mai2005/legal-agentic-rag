from collections.abc import Sequence
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import Modifier, Distance, NamedVector, NamedSparseVector, Filter, PointStruct, VectorParams, SparseVectorParams, SparseVector

class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, api_key: str | None=None) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int, recreate: bool=False) -> None:
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(collection_name=self.collection_name, vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)}, sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)})
            return

        collection = self.client.get_collection(self.collection_name)
        existing_size = collection.config.params.vectors["dense"].size

        if existing_size != vector_size:
            raise ValueError(f"vector size mismatch collection={existing_size}, model={vector_size}")

    def upsert(self, ids: Sequence[str], dense_vectors: Sequence[list[float]], sparse_vectors: Sequence[SparseVector], payloads: Sequence[dict[str, Any]], wait: bool=True) -> None:
        if not (len(ids) == len(dense_vectors) == len(sparse_vectors) == len(payloads)):
            raise ValueError("ids, dense_vectors, sparse_vectors and payloads must have the same length")

        points = [
            PointStruct(id=point_id, vector={"dense": dense_vector, "sparse": sparse_vector}, payload=payload)
            for point_id, dense_vector, sparse_vector, payload in zip(ids, dense_vectors, sparse_vectors, payloads, strict=True)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points, wait=wait)

    def search_dense(self, query_vector: list[float], top_k: int=20, query_filter: Filter | None=None, score_threshold: float | None=None):
        response = self.client.query_points(collection_name=self.collection_name, query=query_vector, using="dense", query_filter=query_filter, limit=top_k, score_threshold=score_threshold, with_payload=True, with_vectors=False)
        return response.points

    def search_sparse(self, query_vector: SparseVector, top_k: int=20, query_filter: Filter | None=None, score_threshold: float | None=None):
        response = self.client.query_points(collection_name=self.collection_name, query=query_vector, using="sparse", query_filter=query_filter, limit=top_k, score_threshold=score_threshold, with_payload=True, with_vectors=False)
        return response.points

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def collection_info(self):
        return self.client.get_collection(self.collection_name)

    def delete(self, ids: Sequence[str], wait: bool=True) -> None:
        self.client.delete(collection_name=self.collection_name, points_selector=list(ids), wait=wait)