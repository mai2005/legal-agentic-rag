import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from app.api.query_routes import router as query_router
from src.llm.llm_client import LLMClient
from src.embedding.embedding_client import BGEEmbeddingClient
from src.storage.qdrant_store import QdrantVectorStore
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("legal_rag_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo environment variables
    load_dotenv()
    
    logger.info("Đang khởi tạo hệ thống Legal Agentic RAG...")
    
    # 1. Khởi tạo LLM Client
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    
    llm_client = LLMClient(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
        prompts_dir="src/prompts"
    )
    app.state.llm_client = llm_client
    
    # 2. Khởi tạo Embedding & Vector Store
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080")
    sparse_embedding_base_url = os.getenv("SPARSE_EMBEDDING_BASE_URL", None)
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "legal_chunks_bge_m3")
    reranker_base_url = os.getenv("RERANKER_BASE_URL", "http://localhost:8081")
    
    try:
        embedding_client = BGEEmbeddingClient(
            base_url=embedding_base_url,
            sparse_base_url=sparse_embedding_base_url
        )
        vector_store = QdrantVectorStore(url=qdrant_url, collection_name=qdrant_collection)
        vector_retriever = VectorRetriever(vector_store=vector_store, embedding_client=embedding_client)
        hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever)
        app.state.vector_store = vector_store
    except Exception as e:
        logger.warning(f"Không thể kết nối trực tiếp VectorStore/Embedding: {e}. Sử dụng cấu hình fallback.")
        vector_retriever = None
        hybrid_retriever = None
        app.state.vector_store = None

    # 3. Khởi tạo Reranker Client & Mock Graph Retriever
    try:
        reranker_client = BGERerankerClient(base_url=reranker_base_url)
    except Exception:
        reranker_client = None

    graph_retriever = GraphRetriever()

    # 4. Khởi tạo Orchestrator Nodes & Compile LangGraph
    config = OrchestratorConfig(
        max_sub_queries=int(os.getenv("MAX_SUB_QUERIES", "4")),
        vector_top_k=int(os.getenv("VECTOR_TOP_K", "20")),
        graph_top_k=int(os.getenv("GRAPH_TOP_K", "20")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "8")),
        min_evidence_score=float(os.getenv("MIN_EVIDENCE_SCORE", "0.5")),
        max_retries=int(os.getenv("MAX_RETRIES", "2")),
    )

    compiled_graph = build_graph(
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

    app.state.graph = compiled_graph
    logger.info("Agentic RAG pipeline đã sẵn sàng phục vụ.")

    yield

    # Cleanup khi shutdown
    logger.info("Đang tắt hệ thống...")
    if hasattr(app.state, "llm_client") and app.state.llm_client:
        app.state.llm_client.close()

app = FastAPI(
    title="Legal Agentic RAG API",
    description="Hệ thống Agentic RAG tư vấn và tra cứu văn bản quy phạm pháp luật Việt Nam.",
    version="1.0.0",
    lifespan=lifespan
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký routes
app.include_router(query_router)

@app.get("/", summary="Root endpoint")
async def root():
    return {
        "name": "Legal Agentic RAG API",
        "status": "online",
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
