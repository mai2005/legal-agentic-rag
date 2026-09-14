import re
import logging
from typing import Any
from src.storage.neo4j_store import Neo4jStore
from src.graph.cypher_templates import (
    QUERY_MULTI_HOP_REFERENCES,
    QUERY_INCOMING_REFERENCES,
    QUERY_GUIDING_DOCUMENTS,
    QUERY_LEGAL_EVOLUTION,
    QUERY_ARTICLE_HIERARCHY,
    QUERY_FIND_ARTICLE_BY_NUMBER,
    QUERY_ARTICLES_BY_CONCEPT,
)
from src.graph.context_builder import GraphContextBuilder

logger = logging.getLogger("graph_retriever")

class GraphRetriever:

    ARTICLE_LOOKUP_PATTERN = re.compile(
        r"(?:Điều\s+(?P<art_num>\d+[a-zA-Z]?))(?:\s+(?:của|thuộc)?\s*(?P<doc_name>[A-ZÀ-Ỹa-zà-ỹ0-9\s\-]+))?",
        re.IGNORECASE,
    )

    def __init__(self, neo4j_store: Neo4jStore | None = None) -> None:
        self.store = neo4j_store

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.store:
            return []

        results: list[dict[str, Any]] = []
        seen_article_ids: set[str] = set()

        match = self.ARTICLE_LOOKUP_PATTERN.search(query)
        if match:
            art_num = match.group("art_num")
            doc_name = (match.group("doc_name") or "").strip()
            
            find_res = self.store.execute_query(
                QUERY_FIND_ARTICLE_BY_NUMBER,
                {
                    "doc_query": doc_name or query,
                    "doc_id": doc_name,
                    "article_number": art_num,
                }
            )

            for item in find_res:
                art_id = item.get("article_id")
                if art_id and art_id not in seen_article_ids:
                    seen_article_ids.add(art_id)
                    results.append(self._format_result_item(item, score=0.95))

                    related = self.get_related_articles(art_id, limit=top_k // 2)
                    for rel_item in related:
                        rel_id = rel_item.get("article_id")
                        if rel_id and rel_id not in seen_article_ids:
                            seen_article_ids.add(rel_id)
                            results.append(rel_item)

        concept_res = self.store.execute_query(
            QUERY_ARTICLES_BY_CONCEPT,
            {
                "concept_query": query,
                "limit": max(3, top_k // 2),
            }
        )
        for item in concept_res:
            art_id = item.get("article_id")
            if art_id and art_id not in seen_article_ids:
                seen_article_ids.add(art_id)
                results.append(self._format_result_item(item, score=0.88))

        return results[:top_k]

    def get_related_articles(self, article_id: str, limit: int = 10) -> list[dict[str, Any]]:
        if not self.store or not article_id:
            return []

        raw_results = self.store.execute_query(
            QUERY_MULTI_HOP_REFERENCES,
            {"article_id": article_id, "limit": limit}
        )

        formatted = []
        for r in raw_results:
            formatted.append(self._format_result_item(r, score=0.80))
        return formatted

    def get_guiding_documents(self, doc_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.store or not doc_id:
            return []

        return self.store.execute_query(
            QUERY_GUIDING_DOCUMENTS,
            {"doc_id": doc_id, "limit": limit}
        )

    def get_legal_evolution(self, doc_id: str) -> list[dict[str, Any]]:
        if not self.store or not doc_id:
            return []

        return self.store.execute_query(
            QUERY_LEGAL_EVOLUTION,
            {"doc_id": doc_id}
        )

    def get_article_hierarchy(self, article_id: str) -> dict[str, Any] | None:
        if not self.store or not article_id:
            return None

        res = self.store.execute_query(
            QUERY_ARTICLE_HIERARCHY,
            {"article_id": article_id}
        )
        return res[0] if res else None

    @staticmethod
    def _format_result_item(item: dict[str, Any], score: float = 0.85) -> dict[str, Any]:
        art_id = item.get("article_id", "")
        return {
            "chunk_id": f"graph_{art_id}",
            "doc_id": item.get("document_id") or item.get("doc_id", ""),
            "document_id": item.get("document_id") or item.get("doc_id", ""),
            "document_title": item.get("document_title") or item.get("title", ""),
            "document_type": item.get("document_type", ""),
            "article_id": art_id,
            "article_number": str(item.get("article_number", "")),
            "article_title": item.get("article_title", ""),
            "chapter_number": item.get("chapter_number"),
            "chapter_title": item.get("chapter_title"),
            "content": item.get("content", ""),
            "score": score,
            "source": "graph",
            "metadata": {
                "source_split": "knowledge_graph",
                "hop_distance": item.get("hop_distance", 1),
                "concept_name": item.get("concept_name"),
            }
        }