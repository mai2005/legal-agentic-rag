from typing import Any

class GraphRetrievalNode:
    def __init__(self, graph_store=None, top_k: int = 20):
        self.graph_store = graph_store
        self.top_k = top_k

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        if not self.graph_store or not hasattr(self.graph_store, "search"):
            return {"graph_results": []}

        seen_chunks: set[str] = set()

        for item in state.get("sub_queries", []):
            route = state.get("retrieval_routes", {}).get(item["id"])
            if route not in ("graph", "hybrid"):
                continue
            query = item.get("rewritten_query", item["query"])
            graph_results = self.graph_store.search(query=query, top_k=self.top_k)
            for res in graph_results:
                cid = res.get("chunk_id") or res.get("article_id")
                if cid and cid not in seen_chunks:
                    seen_chunks.add(cid)
                    results.append(res)

        entities = state.get("query_entities", [])
        if entities and isinstance(entities, list):
            for ent in entities[:3]:
                if isinstance(ent, str) and len(ent.strip()) >= 3:
                    ent_results = self.graph_store.search(query=ent.strip(), top_k=max(2, self.top_k // 4))
                    for res in ent_results:
                        cid = res.get("chunk_id") or res.get("article_id")
                        if cid and cid not in seen_chunks:
                            seen_chunks.add(cid)
                            results.append(res)

        return {"graph_results": results[: self.top_k]}
