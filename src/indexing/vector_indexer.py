import uuid
from typing import Any
from src.storage.qdrant_store import QdrantVectorStore
from src.embedding.embedding_client import BGEEmbeddingClient

class VectorIndexer:
    def __init__(self, embedding_client: BGEEmbeddingClient, vector_store: QdrantVectorStore, upsert_batch_size: int=128) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.upsert_batch_size = upsert_batch_size

    def index(self, chunks: list[dict[str, Any]], recreate_collection: bool = True) -> int:
        valid_chunks = [chunk for chunk in chunks if chunk.get("embedding_text", "").strip()]

        if not valid_chunks:
            return 0

        sample_embedding = self.embedding_client.embed_query(valid_chunks[0]["embedding_text"])

        self.vector_store.ensure_collection(vector_size=len(sample_embedding.dense), recreate=recreate_collection)

        indexed_count = 0

        for start in range(0, len(valid_chunks), self.upsert_batch_size):
            batch = valid_chunks[start: start+self.upsert_batch_size]
            texts = [chunk["embedding_text"] for chunk in batch]
            embeddings = self.embedding_client.embed_document(texts)

            dense_vectors = [embedding.dense for embedding in embeddings]
            sparse_vectors = [embedding.sparse for embedding in embeddings]

            ids: list[str] = []
            payloads: list[dict[str, Any]] = []

            for chunk in batch:
                ids.append(self._build_point_id(chunk))
                payloads.append(self._build_payload(chunk))

            self.vector_store.upsert(ids=ids, dense_vectors=dense_vectors, sparse_vectors=sparse_vectors, payloads=payloads)
            indexed_count += len(batch)
        
        return indexed_count

    @staticmethod
    def _build_point_id(chunk: dict[str, Any]) -> str:
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            raise ValueError("Chunk thieu chunk_id")

        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))

    @staticmethod
    def _build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "document_type": chunk["document_type"],
            "article_id": chunk["article_id"],
            "article_number": chunk["article_number"],
            "article_title": chunk["article_title"],
            "part_number": chunk["part_number"],
            "part_title": chunk["part_title"],
            "chapter_number": chunk["chapter_number"],
            "chapter_title": chunk["chapter_title"],
            "source": chunk["source"],
            "source_split": chunk["source_split"],
            "path": chunk["path"],
            "text": chunk["embedding_text"],
            "content": chunk["content"],
        }