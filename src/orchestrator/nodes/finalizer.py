class Finalizer:
    def run(self, state):
        answer = state.get("verified_answer")
        if not answer:
            answer = state.get("draft_answer", "")
        return {"final_answer": answer}