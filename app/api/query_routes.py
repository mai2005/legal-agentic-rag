import asyncio
from typing import Any
from fastapi import APIRouter, Request, HTTPException, status
from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse, HealthResponse, EvidenceItem, SubQueryItem

router = APIRouter(tags=["Legal Agentic RAG"])

@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái hệ thống")
async def health_check(request: Request) -> HealthResponse:
    graph_ready = hasattr(request.app.state, "graph") and request.app.state.graph is not None
    services = {
        "graph_pipeline": graph_ready,
        "llm_client": hasattr(request.app.state, "llm_client") and request.app.state.llm_client is not None,
        "vector_store": hasattr(request.app.state, "vector_store") and request.app.state.vector_store is not None,
    }
    return HealthResponse(
        status="ok" if graph_ready else "degraded",
        services=services
    )

@router.post("/api/query", response_model=QueryResponse, summary="Tra cứu và trả lời câu hỏi pháp lý")
async def process_legal_query(request_data: QueryRequest, request: Request) -> QueryResponse:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Agentic RAG chưa sẵn sàng. Vui lòng thử lại sau."
        )

    try:
        initial_state: dict[str, Any] = {
            "query": request_data.query,
            "max_retries": request_data.max_retries,
        }

        final_state: dict[str, Any] = await asyncio.to_thread(graph.invoke, initial_state)

        sub_queries_out: list[SubQueryItem] = []
        for sq in final_state.get("sub_queries", []):
            sub_queries_out.append(SubQueryItem(
                id=sq.get("id", ""),
                query=sq.get("query", ""),
                purpose=sq.get("purpose"),
                dependencies=sq.get("dependencies", []),
                rewritten_query=sq.get("rewritten_query")
            ))

        evidence_out: list[EvidenceItem] = []
        for ev in final_state.get("evidence", []):
            evidence_out.append(EvidenceItem(
                chunk_id=ev.get("chunk_id"),
                doc_id=ev.get("doc_id") or ev.get("document_id"),
                document_title=ev.get("document_title"),
                article_number=ev.get("article_number"),
                article_title=ev.get("article_title"),
                content=ev.get("content", ""),
                score=float(ev.get("score", 0.0)),
                source=ev.get("source", "vector")
            ))

        return QueryResponse(
            query=final_state.get("query", request_data.query),
            query_type=final_state.get("query_type"),
            query_entities=final_state.get("query_entities", []),
            final_answer=final_state.get("final_answer", ""),
            verified_answer=final_state.get("verified_answer"),
            verification_passed=bool(final_state.get("verification_passed", False)),
            verification_issues=final_state.get("verification_issues", []),
            evidence_valid=bool(final_state.get("evidence_valid", False)),
            evidence_score=float(final_state.get("evidence_score", 0.0)),
            sub_queries=sub_queries_out,
            dependency_plan=final_state.get("dependency_plan", {}),
            evidence=evidence_out,
            retry_count=int(final_state.get("retry_count", 0)),
            execution_trace=final_state.get("execution_trace", [])
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình xử lý câu hỏi pháp lý: {str(e)}"
        ) from e
