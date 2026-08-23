class QueryDecomposer:
    def __init__(self, llm, max_sub_queries: int=4):
        self.llm = llm
        self.max_sub_queries = max_sub_queries

    def run(self, state):
        query = state["query"]
        sub_queries = self.llm.decompose_query(query=query, max_sub_queries=self.max_sub_queries)
        return {"sub_queries": sub_queries}