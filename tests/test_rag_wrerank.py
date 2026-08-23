import os
import sys
import json
import argparse
import atexit
from typing import Any, List, Set, Dict

from qdrant_client import QdrantClient
from src.storage.qdrant_store import QdrantVectorStore
from src.embedding.embedding_client import BGEEmbeddingClient
from src.rag.vector_retriever import VectorRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.models import RetrievedChunk
from src.rag.reranker import BGERerankerClient


def load_env(env_path: str = ".env") -> Dict[str, str]:
    """Tải cấu hình từ file .env thủ công để tránh phụ thuộc thư viện python-dotenv."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, val = line.split("=", 1)
                        env_vars[key.strip()] = val.strip()
    return env_vars


def calculate_metrics(retrieved_chunks: List[RetrievedChunk], ground_truth: Set[str], top_k: int) -> Dict[str, float]:
    """Tính toán Precision (Actual) và Recall ở cấp độ document_id.
    
    Định nghĩa:
      - R (Retrieved Unique Docs): Tập hợp các document_id duy nhất được trả về từ top_k chunks.
      - G (Ground Truth Docs): Tập hợp các document_id chính xác từ answer.
      - Precision = |R ∩ G| / |R|
      - Recall@k = |R ∩ G| / |G|
    """
    # Lấy danh sách document_id duy nhất từ top_k chunks đầu tiên (giữ nguyên thứ tự rank)
    retrieved_docs = []
    for chunk in retrieved_chunks[:top_k]:
        if chunk.document_id and chunk.document_id not in retrieved_docs:
            retrieved_docs.append(chunk.document_id)

    # Lấy tập hợp
    retrieved_set = set(retrieved_docs)
    intersection = retrieved_set.intersection(ground_truth)
    
    # Precision thực tế (chia cho số lượng doc_id duy nhất)
    precision = len(intersection) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    
    # Recall@k (Số tài liệu đúng đã tìm được chia cho tổng số tài liệu đúng)
    if len(ground_truth) > 0:
        recall = len(intersection) / len(ground_truth)
    else:
        recall = 1.0  # Mặc định là 1.0 nếu tập ground truth rỗng

    return {
        "precision": precision,
        "recall_at_k": recall,
        "unique_retrieved_count": len(retrieved_set),
        "hit": 1.0 if len(intersection) > 0 else 0.0
    }


class Tee(object):
    """Lớp hỗ trợ ghi đồng thời ra màn hình (stdout) và file log."""
    def __init__(self, filename: str, mode: str = "a", encoding: str = "utf-8"):
        self.file = open(filename, mode, encoding=encoding)
        self.stdout = sys.stdout

    def write(self, message: str) -> None:
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng truy vấn RAG (Precision & Recall).")
    parser.add_argument("--data_path", type=str, default="data/train.json", help="Đường dẫn tới file train.json")
    parser.add_argument("--limit", type=int, default=1000, help="Giới hạn số câu hỏi chạy thử nghiệm (để trống hoặc đặt <= 0 để chạy hết)")
    parser.add_argument("--top_k", type=int, default=20, help="Số lượng kết quả cần truy vấn (top_k)")
    parser.add_argument("--retriever_type", type=str, default="vector", choices=["hybrid", "vector"], help="Loại retriever cần kiểm thử")
    parser.add_argument("--verbose", action="store_false", help="Hiển thị chi tiết từng câu hỏi lỗi hoặc kết quả")
    
    args = parser.parse_args()

    # Thiết lập ghi log song song ra file log_metric
    logger = Tee("log_metric", mode="a", encoding="utf-8")
    sys.stdout = logger
    atexit.register(logger.close)

    # Load environment variables
    env = load_env()
    qdrant_url = env.get("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = env.get("QDRANT_COLLECTION", "legal_chunks_bge_m3")
    embedding_base_url = env.get("EMBEDDING_BASE_URL", "http://localhost:8080")
    # sparse_embedding_base_url = env.get("SPARSE_EMBEDDING_BASE_URL", "http://localhost:8082")
    sparse_embedding_base_url = None
    reranker_base_url = env.get("RERANKER_BASE_URL", "http://localhost:8081")
    
    print("=" * 70)
    print("THÔNG TIN CẤU HÌNH HỆ THỐNG (CÓ RERANKER)")
    print("=" * 70)
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Collection: {qdrant_collection}")
    print(f"Embedding URL: {embedding_base_url}")
    print(f"Reranker URL: {reranker_base_url}")
    print(f"Retriever chọn: {args.retriever_type.upper()}")
    print(f"Top K yêu cầu: {args.top_k}")
    print("=" * 70)

    # Đọc dữ liệu train.json
    if not os.path.exists(args.data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {args.data_path}")
        return

    with open(args.data_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Lọc danh sách test cases
    test_cases = []
    for key, val in train_data.items():
        question = val.get("question", "")
        answers = val.get("answer", [])
        if question and answers:
            # answers có thể là list các string document_id
            test_cases.append({
                "id": key,
                "query": question,
                "ground_truth": set(str(ans) for ans in answers)
            })

    total_available = len(test_cases)
    if args.limit > 0 and args.limit < total_available:
        test_cases = test_cases[:args.limit]
    
    print(f"Tổng số câu hỏi trong file gốc: {total_available}")
    print(f"Số câu hỏi sẽ được đánh giá: {len(test_cases)}")
    print("=" * 70)

    # Khởi tạo VectorStore và Clients
    vector_store = QdrantVectorStore(url=qdrant_url, collection_name=qdrant_collection)
    
    # Khởi tạo embedding client, reranker client và thực hiện đánh giá
    with BGEEmbeddingClient(
        base_url=embedding_base_url,
        sparse_base_url=sparse_embedding_base_url,
        batch_size=32,
        timeout=60.0,
        normalize=True,
    ) as embedding_client, BGERerankerClient(
        base_url=reranker_base_url,
        timeout=60.0,
    ) as reranker_client:
        
        vector_retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_client=embedding_client
        )
        
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            reranker=reranker_client
        )

        # Biến thống kê
        total_precision_5 = 0.0
        total_recall_5 = 0.0
        total_recall_10 = 0.0
        total_recall_15 = 0.0
        total_recall_20 = 0.0
        total_hit_rate_5 = 0.0
        total_hit_rate_10 = 0.0
        total_hit_rate_15 = 0.0
        total_hit_rate_20 = 0.0
        
        count_top_k_correct = 0
        count_processed = 0

        print("Đang bắt đầu truy vấn và tính toán chỉ số...")
        print("-" * 70)

        for idx, tc in enumerate(test_cases, 1):
            query = tc["query"]
            gt = tc["ground_truth"]

            try:
                # Thực hiện truy vấn
                if args.retriever_type == "hybrid":
                    # HybridRetriever hỗ trợ hàm retrieve trực tiếp
                    results = hybrid_retriever.retrieve(query=query, top_k=args.top_k)
                else:
                    # VectorRetriever (Dense Search)
                    embedding = vector_retriever.embed_query(query)
                    results = vector_retriever.dense_search(embedding, query_filter=None, top_k=args.top_k)

                # 1. Kiểm tra xem số lượng kết quả trả về có hợp lệ không
                is_top_k_correct = len(results) <= args.top_k
                if is_top_k_correct:
                    count_top_k_correct += 1

                # 2. Tính toán các chỉ số tại K = 5, 10, 15, 20
                metrics_5 = calculate_metrics(results, gt, 5)
                metrics_10 = calculate_metrics(results, gt, 10)
                metrics_15 = calculate_metrics(results, gt, 15)
                metrics_20 = calculate_metrics(results, gt, 20)

                total_precision_5 += metrics_5["precision"]
                total_recall_5 += metrics_5["recall_at_k"]
                total_recall_10 += metrics_10["recall_at_k"]
                total_recall_15 += metrics_15["recall_at_k"]
                total_recall_20 += metrics_20["recall_at_k"]
                total_hit_rate_5 += metrics_5["hit"]
                total_hit_rate_10 += metrics_10["hit"]
                total_hit_rate_15 += metrics_15["hit"]
                total_hit_rate_20 += metrics_20["hit"]
                
                count_processed += 1

                if args.verbose:
                    print(f"[{idx}/{len(test_cases)}] QID: {tc['id']}")
                    print(f"  Query: {query[:80]}...")
                    print(f"  Ground Truth Document IDs: {list(gt)}")
                    ret_ids = [chunk.document_id for chunk in results[:20]]
                    print(f"  Retrieved Chunk Document IDs (top 20): {ret_ids}")
                    print(
                        f"  Recall@5: {metrics_5['recall_at_k']:.4f} | "
                        f"Recall@10: {metrics_10['recall_at_k']:.4f} | "
                        f"Recall@15: {metrics_15['recall_at_k']:.4f} | "
                        f"Recall@20: {metrics_20['recall_at_k']:.4f}"
                    )
                    print(f"  Query limit check (<= {args.top_k}): {'PASS' if is_top_k_correct else 'FAIL'}")
                    print("-" * 50)

                # In tiến trình mỗi 10%
                if not args.verbose and idx % max(1, len(test_cases) // 10) == 0:
                    print(f"Đã xử lý {idx}/{len(test_cases)} câu hỏi...")

            except Exception as e:
                print(f"Lỗi khi xử lý câu hỏi ID {tc['id']}: {e}")

        # Tính toán điểm trung bình
        if count_processed > 0:
            mean_precision_5 = total_precision_5 / count_processed
            mean_recall_5 = total_recall_5 / count_processed
            mean_recall_10 = total_recall_10 / count_processed
            mean_recall_15 = total_recall_15 / count_processed
            mean_recall_20 = total_recall_20 / count_processed
            mean_hit_rate_5 = total_hit_rate_5 / count_processed
            mean_hit_rate_10 = total_hit_rate_10 / count_processed
            mean_hit_rate_15 = total_hit_rate_15 / count_processed
            mean_hit_rate_20 = total_hit_rate_20 / count_processed
            top_k_pass_rate = count_top_k_correct / count_processed
        else:
            mean_precision_5 = 0.0
            mean_recall_5 = mean_recall_10 = mean_recall_15 = mean_recall_20 = 0.0
            mean_hit_rate_5 = mean_hit_rate_10 = mean_hit_rate_15 = mean_hit_rate_20 = 0.0
            top_k_pass_rate = 0.0

        # Xuất báo cáo đánh giá
        print("\n" + "=" * 70)
        print("KẾT QUẢ ĐÁNH GIÁ PIPELINE RETRIEVAL")
        print("=" * 70)
        print(f"Tổng số câu hỏi đã đánh giá thành công: {count_processed}")
        print(f"Số lượng truy vấn yêu cầu (Top K): {args.top_k}")
        print("-" * 70)
        
        # 1. Kết quả kiểm tra truy vấn top K
        print(f"XÁC MINH SỐ LƯỢNG TRUY VẤN (TOP {args.top_k}):")
        if top_k_pass_rate == 1.0:
            print(f"  => [PASS] Tất cả các truy vấn đều trả về tối đa {args.top_k} kết quả (Đạt 100%).")
        else:
            print(f"  => [FAIL] Có {count_processed - count_top_k_correct} câu hỏi trả về nhiều hơn {args.top_k} kết quả.")
            print(f"  => Tỷ lệ đúng giới hạn số lượng: {top_k_pass_rate * 100:.2f}%")
        
        # 2. Kết quả Precision & Recall & Hit Rate
        print("-" * 70)
        print("CHỈ SỐ ĐÁNH GIÁ CHẤT LƯỢNG (METRICS):")
        print(f"  - Mean Precision@5       : {mean_precision_5 * 100:.2f}%")
        print(f"  - Mean Recall@5          : {mean_recall_5 * 100:.2f}%")
        print(f"  - Mean Recall@10         : {mean_recall_10 * 100:.2f}%")
        print(f"  - Mean Recall@15         : {mean_recall_15 * 100:.2f}%")
        print(f"  - Mean Recall@20         : {mean_recall_20 * 100:.2f}%")
        print(f"  - Hit Rate@5             : {mean_hit_rate_5 * 100:.2f}%")
        print(f"  - Hit Rate@10            : {mean_hit_rate_10 * 100:.2f}%")
        print(f"  - Hit Rate@15            : {mean_hit_rate_15 * 100:.2f}%")
        print(f"  - Hit Rate@20            : {mean_hit_rate_20 * 100:.2f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
