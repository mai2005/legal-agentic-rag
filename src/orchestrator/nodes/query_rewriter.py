class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        rewritten = []
        for item in state["sub_queries"]:
            query = item["query"]
            new_query = self.llm.rewrite_query(
                query=query,
                query_type=state.get("query_type"),
                entities=state.get("query_entities"),
                metadata=state.get("query_metadata"),
                dependency_context=item.get("purpose")
            )
            item = dict(item)
            item["rewritten_query"] = new_query
            rewritten.append(item)
        return {"sub_queries": rewritten}