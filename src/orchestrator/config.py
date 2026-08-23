from dataclasses import dataclass

@dataclass(frozen=True)
class OrchestratorConfig:
    max_sub_queries: int = 4
    
    vector_top_k: int = 20
    graph_top_k: int = 20
    rerank_top_k: int = 8

    min_evidence_score: float = 0.5
    max_retries: int = 2

    enable_graph: bool = True
    enable_verification: bool = True