class Answerer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        answer = self.llm.generate_answer(
            query=state["query"],
            evidence=state["evidence"],
            query_type=state.get("query_type"),
            metadata=state.get("query_metadata")
        )
        return {"draft_answer": answer}