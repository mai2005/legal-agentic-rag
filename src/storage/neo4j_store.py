import logging
from typing import Any
from collections.abc import Sequence
from neo4j import GraphDatabase, Driver

logger = logging.getLogger("neo4j_store")

class Neo4jStore:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        max_connection_pool_size: int = 50,
    ) -> None:
        self.uri = uri
        self.database = database
        self.driver: Driver = GraphDatabase.driver(
            self.uri,
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
        )

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def verify_connectivity(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Không thể kết nối Neo4j tại {self.uri}: {e}")
            return False

    def ensure_constraints_and_indexes(self) -> None:
        constraints = [
            "CREATE CONSTRAINT document_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT article_article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT chapter_chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE",
            "CREATE CONSTRAINT part_part_id IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (lc:LegalConcept) REQUIRE lc.name IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX article_number_idx IF NOT EXISTS FOR (a:Article) ON (a.article_number)",
            "CREATE INDEX document_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
            "CREATE INDEX concept_normalized_idx IF NOT EXISTS FOR (lc:LegalConcept) ON (lc.normalized_name)",
        ]

        with self.driver.session(database=self.database) as session:
            for statement in constraints:
                try:
                    session.run(statement)
                except Exception as e:
                    logger.warning(f"Lỗi tạo constraint ({statement}): {e}")

            for statement in indexes:
                try:
                    session.run(statement)
                except Exception as e:
                    logger.warning(f"Lỗi tạo index ({statement}): {e}")

    def execute_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        parameters = parameters or {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def batch_upsert_nodes(
        self,
        label: str,
        key_property: str,
        items: Sequence[dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        if not items:
            return 0

        cypher = f"""
        UNWIND $batch AS item
        MERGE (n:`{label}` {{{key_property}: item.{key_property}}})
        SET n += item
        """

        total_upserted = 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                session.run(cypher, {"batch": batch})
                total_upserted += len(batch)

        return total_upserted

    def batch_upsert_relationships(
        self,
        rel_type: str,
        from_label: str,
        from_key: str,
        to_label: str,
        to_key: str,
        rels: Sequence[dict[str, Any]],
        batch_size: int = 1000,
    ) -> int:
        if not rels:
            return 0

        cypher = f"""
        UNWIND $batch AS rel
        MATCH (a:`{from_label}` {{{from_key}: rel.from_id}})
        MATCH (b:`{to_label}` {{{to_key}: rel.to_id}})
        MERGE (a)-[r:`{rel_type}`]->(b)
        SET r += coalesce(rel.properties, {{}})
        """

        total_upserted = 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rels), batch_size):
                batch = rels[i : i + batch_size]
                session.run(cypher, {"batch": batch})
                total_upserted += len(batch)

        return total_upserted

    def count_nodes(self, label: str | None = None) -> int:
        query = f"MATCH (n:`{label}`) RETURN count(n) AS cnt" if label else "MATCH (n) RETURN count(n) AS cnt"
        res = self.execute_query(query)
        return res[0]["cnt"] if res else 0

    def count_relationships(self, rel_type: str | None = None) -> int:
        query = f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS cnt" if rel_type else "MATCH ()-[r]->() RETURN count(r) AS cnt"
        res = self.execute_query(query)
        return res[0]["cnt"] if res else 0

    def clear_database(self) -> None:
        """
        Xóa toàn bộ nodes và relationships trong đồ thị (Cẩn trọng khi dùng!).
        """
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
