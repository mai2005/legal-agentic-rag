from ..routing import route_query

class RetrievalRouter:
    def run(self, state):
        routes = {}
        for item in state["sub_queries"]:
            route = route_query(
                query_type=state.get("query_type", "semantic"),
                has_relation=(state.get("query_type") == "legal_relation"),
                has_semantic_info=True
            )
            routes[item["id"]] = route.value
        return {"retrieval_routes": routes}