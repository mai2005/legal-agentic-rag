from typing import Any, Literal, TypedDict

RetrievalRoute = Literal["vector", "graph", "hybrid"]

class SubQuery(TypedDict, total=False):
    id: str
    query: str
    purpose: str
    dependencies: list[str]
    route: RetrievalRoute
    rewritten_query: str

class RetrievalResult(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any]

class GraphResult(TypedDict, total=False):
    node_id: str
    relation: str
    source: str
    target: str
    metadata: dict[str, Any]

class AgentState(TypedDict, total=False):
    query: str

    query_type: str
    query_entities: list[str]
    query_metadata: dict[str, Any]

    sub_queries: list[SubQuery]
    dependency_plan: dict[str, list[str]]

    retrieval_routes: dict[str, RetrievalRoute]

    vector_results: list[RetrievalResult]
    graph_results: list[GraphResult]

    fused_results: list[RetrievalResult]
    reranked_results: list[RetrievalResult]

    evidence: list[RetrievalResult]
    evidence_valid: bool
    evidence_score: float
    missing_info: list[str]

    draft_answer: str
    verified_answer: str
    final_answer: str

    verification_passed: bool
    verification_issues: list[str]

    retry_count: int
    max_retries: int

    execution_trace: list[dict[str, Any]]