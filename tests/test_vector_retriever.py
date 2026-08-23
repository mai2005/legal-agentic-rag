from src.storage.qdrant_store import QdrantVectorStore
from src.embedding.embedding_client import BGEEmbeddingClient
from src.rag.vector_retriever import VectorRetriever


def main() -> None:
    # Khởi tạo QdrantVectorStore tương tự như các file khác trong hệ thống
    vector_store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="legal_chunks_bge_m3",
    )

    with BGEEmbeddingClient(
        base_url="http://localhost:8080",
        sparse_base_url="http://localhost:8082",
        batch_size=32,
        timeout=60.0,
        max_retries=3,
        normalize=True,
    ) as embedding_client:
        
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_client=embedding_client,
        )

        query = "Thời gian thử việc tối đa đối với người làm công việc cần trình độ đại học là bao nhiêu ngày?"
        print(f"Đang thực hiện tìm kiếm dense vector cho câu hỏi: '{query}'...")
        
        # Nhúng câu truy vấn và thực hiện tìm kiếm dense
        query_embedding = retriever.embed_query(query)
        results = retriever.dense_search(
            embedding=query_embedding,
            query_filter=None,
            top_k=5,
        )

        print(f"Đã tìm thấy {len(results)} kết quả:")
        for rank, result in enumerate(results, start=1):
            print("=" * 80)
            print(f"Rank: {rank}")
            print(f"Score: {result.score:.6f}")
            print(f"Chunk_id: {result.chunk_id}")
            print(f"Điều: {result.article_number}")
            print(f"Tiêu đề: {result.article_title}")
            print(f"Chương: {result.chapter_title}")
            print(f"Phần: {result.part_title}")
            print(f"Văn bản: {result.document_title}")
            print("\nNội dung chunk:")
            print(result.content)


if __name__ == "__main__":
    main()