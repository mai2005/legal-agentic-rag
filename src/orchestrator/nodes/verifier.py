class Verifier:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        result = self.llm.verify_answer(query=state["query"], answer=state["draft_answer"], evidence=state["evidence"])
        return {
            "verification_passed": result["passed"],
            "verification_issues": result.get("issues", []),
            "verified_answer": (state["draft_answer"] if result["passed"] else "")
        }