class FusionNode:
    def run(self, state):
        vector_results = state.get("vector_results", [])
        graph_results = state.get("graph_results", [])
        fused = []
        fused.extend(vector_results)
        for item in graph_results:
            fused.append({**item, "source": "graph"})
        return {"fused_results": self._deduplicate(fused)}

    def _deduplicate(self, results):
        seen = set()
        output = []
        for result in results:
            key = result.get("chunk_id") or result.get("node_id")
            if key in seen:
                continue
            seen.add(key)
            output.append(result)
        return output