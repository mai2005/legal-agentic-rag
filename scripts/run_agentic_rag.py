import os
import sys
import json
import logging
from dotenv import load_dotenv

from src.llm.llm_client import LLMClient
from src.embedding.embedding_client import BGEEmbeddingClient
from src.storage.qdrant_store import QdrantVectorStore
from src.storage.neo4j_store import Neo4jStore
from src.rag.vector_retriever import VectorRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import BGERerankerClient
from src.rag.graph_retriever import GraphRetriever

from src.orchestrator.config import OrchestratorConfig
from src.orchestrator.graph import build_graph
from src.orchestrator.nodes.query_analyzer import QueryAnalyzer
from src.orchestrator.nodes.query_decomposer import QueryDecomposer
from src.orchestrator.nodes.dependency_planner import DependencyPlanner
from src.orchestrator.nodes.query_rewriter import QueryRewriter
from src.orchestrator.nodes.retrieval_router import RetrievalRouter
from src.orchestrator.nodes.vector_retrieval_node import VectorRetrievalNode
from src.orchestrator.nodes.graph_retrieval_node import GraphRetrievalNode
from src.orchestrator.nodes.fusion_node import FusionNode
from src.orchestrator.nodes.reranker_node import RerankerNode
from src.orchestrator.nodes.evidence_validator import EvidenceValidator
from src.orchestrator.nodes.answerer import Answerer
from src.orchestrator.nodes.verifier import Verifier
from src.orchestrator.nodes.finalizer import Finalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agentic_runner")

def main():
    load_dotenv()

    query = sys.argv[1] if len(sys.argv) > 1 else "Điều kiện cấp giấy chứng nhận quyền sử dụng đất theo quy định pháp luật?"
    
    logger.info(f"Đang xử lý câu hỏi: '{query}'")

    # 1. Khởi tạo LLM Client
    llm_client = LLMClient(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        model=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
        prompts_dir="src/prompts"
    )

    # 2. Khởi tạo Vector Retrieval & Reranker
    try:
        embedding_client = BGEEmbeddingClient(
            base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080"),
            sparse_base_url=os.getenv("SPARSE_EMBEDDING_BASE_URL", None)
        )
        vector_store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "legal_chunks_bge_m3")
        )
        vector_retriever = VectorRetriever(vector_store=vector_store, embedding_client=embedding_client)
        hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever)
    except Exception as e:
        logger.warning(f"Không thể kết nối VectorStore/Embedding ({e}). Sử dụng mock.")
        hybrid_retriever = None

    try:
        reranker_client = BGERerankerClient(base_url=os.getenv("RERANKER_BASE_URL", "http://localhost:8081"))
    except Exception:
        reranker_client = None

    try:
        neo4j_store = Neo4jStore(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        if not neo4j_store.verify_connectivity():
            logger.warning("Không thể kết nối Neo4j, sử dụng mock GraphRetriever.")
            neo4j_store = None
    except Exception as e:
        logger.warning(f"Lỗi khởi tạo Neo4jStore ({e}). Sử dụng mock.")
        neo4j_store = None

    graph_retriever = GraphRetriever(neo4j_store=neo4j_store)

    # 3. Khởi tạo Orchestrator Nodes & Compile Graph
    config = OrchestratorConfig(
        max_sub_queries=int(os.getenv("MAX_SUB_QUERIES", "4")),
        vector_top_k=int(os.getenv("VECTOR_TOP_K", "20")),
        graph_top_k=int(os.getenv("GRAPH_TOP_K", "20")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "8")),
        min_evidence_score=float(os.getenv("MIN_EVIDENCE_SCORE", "0.5")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
    )

    app_graph = build_graph(
        config=config,
        query_analyzer=QueryAnalyzer(llm=llm_client),
        query_decomposer=QueryDecomposer(llm=llm_client, max_sub_queries=config.max_sub_queries),
        dependency_planner=DependencyPlanner(llm=llm_client),
        query_rewriter=QueryRewriter(llm=llm_client),
        retrieval_router=RetrievalRouter(),
        vector_retrieval=VectorRetrievalNode(hybrid_retriever=hybrid_retriever, top_k=config.vector_top_k),
        graph_retrieval=GraphRetrievalNode(graph_store=graph_retriever, top_k=config.graph_top_k),
        fusion=FusionNode(),
        reranker=RerankerNode(reranker=reranker_client, top_k=config.rerank_top_k),
        evidence_validator=EvidenceValidator(llm=llm_client, min_score=config.min_evidence_score),
        answerer=Answerer(llm=llm_client),
        verifier=Verifier(llm=llm_client),
        finalizer=Finalizer(),
    )

    # 4. Thực thi pipeline
    result = app_graph.invoke({"query": query})

    print("\n" + "="*50)
    print("KẾT QUẢ AGENTIC RAG")
    print("="*50)
    print(f"Câu hỏi: {result.get('query')}")
    print(f"Loại truy vấn: {result.get('query_type')}")
    print(f"Thực thể: {result.get('query_entities')}")
    print(f"\nCâu hỏi con ({len(result.get('sub_queries', []))}):")
    for sq in result.get("sub_queries", []):
        print(f"  - [{sq.get('id')}] {sq.get('query')} (viết lại: {sq.get('rewritten_query')})")
    print(f"\nKế hoạch phụ thuộc: {json.dumps(result.get('dependency_plan'), ensure_ascii=False)}")
    print(f"\nBằng chứng ({len(result.get('evidence', []))}):")
    for ev in result.get("evidence", [])[:3]:
        print(f"  - [{ev.get('document_title')}] Điều {ev.get('article_number')}: {ev.get('content')[:100]}... (Score: {ev.get('score')})")
    print(f"\nĐiểm bằng chứng: {result.get('evidence_score')} (Hợp lệ: {result.get('evidence_valid')})")
    print(f"Kiểm định trả lời: {result.get('verification_passed')} (Vấn đề: {result.get('verification_issues')})")
    print(f"Số lần retry: {result.get('retry_count')}")
    print(f"\n--- CÂU TRẢ LỜI CUỐI CÙNG ---\n{result.get('final_answer')}")
    print("="*50)

if __name__ == "__main__":
    main()
