from enum import Enum

class RetrievalRoute(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"

def route_query(query_type: str, has_relation: bool, has_semantic_info: bool) -> RetrievalRoute:
    if has_relation and has_semantic_info:
        return RetrievalRoute.HYBRID
    if has_relation: 
        return RetrievalRoute.GRAPH
    return RetrievalRoute.VECTOR