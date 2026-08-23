class DependencyPlanner:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        sub_queries = state["sub_queries"]
        dependencies = self.llm.plan_dependencies(sub_queries)
        return {"dependency_plan": dependencies}