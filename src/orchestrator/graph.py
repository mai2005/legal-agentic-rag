from .state import AgentState
from .config import OrchestratorConfig
from typing import Any
from langgraph.graph import END, START, StateGraph

def _init_state(state: AgentState, config: OrchestratorConfig) -> dict[str, Any]:
    return {
        "retry_count": 0,
        "max_retries": config.max_retries,
        "execution_trace": [],
        "vector_results": [],
        "graph_results": [],
        "fused_results": [],
        "reranked_results": [],
        "evidence": [],
        "evidence_valid": False,
        "evidence_score": 0.0,
        "missing_information": [],
        "verification_passed": False,
        "verification_issues": [],
        "draft_answer": "",
        "verified_answer": "",
        "final_answer": "",
    }

def _retry_handler(state: AgentState) -> dict[str, Any]:
    current_retry = state.get("retry_count", 0)
    return {
        "retry_count": current_retry + 1,

        "vector_results": [],
        "graph_results": [],
        "fused_results": [],
        "reranked_results": [],
        "evidence": [],

        "draft_answer": "",
        "verified_answer": "",

        "evidence_valid": False,
        "evidence_score": 0.0,
        "verification_passed": False,
        
        "execution_trace": (state.get("execution_trace", []) 
        + [{
            "node": "retry_handler",
            "retry_count": current_retry+1
        }])
    }

def _evidence_router(state: AgentState) -> str:
    if state.get("evidence_valid", False):
        return "answer"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count < max_retries:
        return "retry"
    
    return "answer"

def _verification_router(state: AgentState) -> str:
    if state.get("verification_passed", False):
        return "finalize"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count < max_retries:
        return "retry"
    
    return "finalize"

def build_graph(*, config: OrchestratorConfig, query_analyzer, query_decomposer, dependency_planner, query_rewriter, retrieval_router, vector_retrieval, graph_retrieval, fusion, reranker, evidence_validator, answerer, verifier, finalizer):
    workflow = StateGraph(AgentState)

    workflow.add_node("initialize", lambda state: _init_state(state, config))

    workflow.add_node("query_analyzer", query_analyzer.run)
    workflow.add_node("query_decomposer", query_decomposer.run)
    workflow.add_node("dependency_planner", dependency_planner.run)
    workflow.add_node("query_rewriter", query_rewriter.run)
    workflow.add_node("retrieval_router", retrieval_router.run)
    workflow.add_node("vector_retrieval", vector_retrieval.run)
    workflow.add_node("graph_retrieval", graph_retrieval.run)
    workflow.add_node("fusion", fusion.run)
    workflow.add_node("reranker", reranker.run)
    workflow.add_node("evidence_validator", evidence_validator.run)
    workflow.add_node("answerer", answerer.run)
    workflow.add_node("verifier", verifier.run)
    workflow.add_node("finalizer", finalizer.run)
    workflow.add_node("retry_handler", _retry_handler)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "query_analyzer")
    workflow.add_edge("query_analyzer", "query_decomposer")
    workflow.add_edge("query_decomposer", "dependency_planner")
    workflow.add_edge("dependency_planner", "query_rewriter")
    workflow.add_edge("query_rewriter", "retrieval_router")
    workflow.add_edge("retrieval_router", "vector_retrieval")
    workflow.add_edge("retrieval_router", "graph_retrieval")
    workflow.add_edge("vector_retrieval", "fusion")
    workflow.add_edge("graph_retrieval", "fusion")
    workflow.add_edge("fusion", "reranker")
    workflow.add_edge("reranker", "evidence_validator")

    workflow.add_conditional_edges("evidence_validator", _evidence_router, {"answer": "answerer", "retry": "retry_handler"})
    workflow.add_edge("answerer", "verifier")
    workflow.add_conditional_edges("verifier", _verification_router, {"finalize": "finalizer", "retry": "retry_handler"})
    workflow.add_edge("retry_handler", "query_rewriter")
    workflow.add_edge("finalizer", END)

    return workflow.compile()