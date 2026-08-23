from typing import Any
class GraphRetrievalNode:
    def __init__(self, graph_store=None, top_k: int=20):
        self.graph_store = graph_store
        self.top_k = top_k

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for item in state.get("sub_queries", []):
            route = state.get("retrieval_routes", {}).get(item["id"])
            if route not in ("graph", "hybrid"):
                continue
            query = item.get("rewritten_query", item["query"])
            if self.graph_store and hasattr(self.graph_store, "search"):
                graph_results = self.graph_store.search(query=query, top_k=self.top_k)
                results.extend(graph_results)
        return {"graph_results": results}