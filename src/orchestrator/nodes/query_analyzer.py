from typing import Any

class QueryAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[dict, Any]:
        query = state["query"]
        analysis = self.llm.analyze_query(query)
        return {
            "query_type": analysis.get("query_type", "semantic"),
            "query_entities": analysis.get("entities", []),
            "query_metadata": analysis.get("metadata", {})
        }