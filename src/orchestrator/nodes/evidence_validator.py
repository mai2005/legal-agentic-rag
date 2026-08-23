class EvidenceValidator:
    def __init__(self, llm=None, min_score: float=0.5):
        self.llm = llm
        self.min_score = min_score

    def run(self, state):
        evidence = state.get("reranked_results", [])
        if not evidence:
            return {
                "evidence": [],
                "evidence_valid": False,
                "missing_info": ["No relevant evidence found"]
            }
        score = self._calculate_score(evidence)
        valid = score >= self.min_score
        return {
            "evidence": evidence,
            "evidence_valid": valid,
            "evidence_score": score,
            "missing_info": [] if valid else ["Evidence relevance is insuddicient"]
        }

    def _calculate_score(self, evidence):
        scores = [item.get("score", 0.0) for item in evidence]
        if not scores:
            return 0.0
        return max(scores)