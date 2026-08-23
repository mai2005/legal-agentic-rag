from src.indexing.vector_indexer import VectorIndexer
from src.embedding.embedding_client import BGEEmbeddingClient
from src.storage.qdrant_store import QdrantVectorStore
from pathlib import Path
import json
import os
from typing import Any

def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("File chunk phai chua mot json array")

    return data

def main() -> None:
    embedding_url = os.getenv(
        "EMBEDDING_BASE_URL",
        "http://localhost:8080",
    )
    sparse_embedding_url = os.getenv(
        "SPARSE_EMBEDDING_BASE_URL",
        "http://localhost:8082",
    )
    qdrant_url = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )
    collection_name = os.getenv(
        "QDRANT_COLLECTION",
        "legal_chunks_bge_m3",
    )
    chunks_path = os.getenv(
        "CHUNKS_PATH",
        "data/processed/chunks.json",
    )

    chunks = load_chunks(chunks_path)

    vector_store = QdrantVectorStore(url=qdrant_url, collection_name=collection_name)
    
    with BGEEmbeddingClient(base_url=embedding_url, sparse_base_url=None, batch_size=32, normalize=True) as embedding_client:
        indexer = VectorIndexer(embedding_client=embedding_client, vector_store=vector_store, upsert_batch_size=128)
        count = indexer.index(chunks=chunks, recreate_collection=True)

    print(f"Da index {count} chunks vao {collection_name}")

if __name__ == "__main__":
    main()
