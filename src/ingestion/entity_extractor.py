import re
import unicodedata
from typing import Any
from src.graph.schema import LegalConceptNode, RelationshipRecord, RelationType

class LegalConceptExtractor:

    DEFINITION_CLAUSE_PATTERN = re.compile(
        r"^\s*(?:\d+[\.\)]\s*)?(?:[\"“'‘](?P<concept_quoted>[^\"”'’]+)[\"”'’]|(?P<concept_plain>[A-ZÀ-Ỹa-zà-ỹ\s]{3,50}))\s+(?:là|được\s+hiểu\s+là|bao\s+gồm)\s+(?P<desc>.*)",
        re.MULTILINE,
    )

    def __init__(self) -> None:
        self.seen_concepts: dict[str, LegalConceptNode] = {}

    @staticmethod
    def slugify(text: str) -> str:
        if not text:
            return ""
        val = unicodedata.normalize("NFD", text)
        val = "".join(c for c in val if unicodedata.category(c) != "Mn")
        val = val.lower().strip()
        val = re.sub(r"[^a-z0-9]+", "_", val)
        return val.strip("_")

    def extract_from_article(
        self,
        article_id: str,
        article_title: str | None,
        article_content: str,
    ) -> tuple[list[LegalConceptNode], list[RelationshipRecord]]:
        concept_nodes: list[LegalConceptNode] = []
        records: list[RelationshipRecord] = []

        is_definition_article = bool(article_title and re.search(r"giải\s+thích\s+từ\s+ngữ|định\s+nghĩa", article_title, re.IGNORECASE))
        
        if is_definition_article:
            for match in self.DEFINITION_CLAUSE_PATTERN.finditer(article_content):
                name = (match.group("concept_quoted") or match.group("concept_plain") or "").strip()
                desc = match.group("desc").strip()
                
                if len(name) < 2 or len(name) > 80:
                    continue

                norm_name = self.slugify(name)
                concept_id = f"concept_{norm_name}"

                node = LegalConceptNode(
                    concept_id=concept_id,
                    name=name,
                    normalized_name=norm_name,
                    description=desc[:500],
                    category="khái niệm",
                )
                concept_nodes.append(node)
                self.seen_concepts[norm_name] = node

                records.append(
                    RelationshipRecord(
                        from_id=article_id,
                        to_id=concept_id,
                        rel_type=RelationType.DEFINES,
                        from_label="Article",
                        to_label="LegalConcept",
                        properties={"term": name},
                    )
                )

        if article_title and not is_definition_article:
            clean_title = re.sub(r"^(?:quy\s+định\s+về|về)\s+", "", article_title, flags=re.IGNORECASE).strip()
            if 3 <= len(clean_title) <= 50 and not clean_title.isdigit():
                norm_title = self.slugify(clean_title)
                concept_id = f"concept_{norm_title}"

                node = LegalConceptNode(
                    concept_id=concept_id,
                    name=clean_title,
                    normalized_name=norm_title,
                    description=None,
                    category="chủ đề",
                )
                concept_nodes.append(node)
                self.seen_concepts[norm_title] = node

                records.append(
                    RelationshipRecord(
                        from_id=article_id,
                        to_id=concept_id,
                        rel_type=RelationType.REGULATES,
                        from_label="Article",
                        to_label="LegalConcept",
                        properties={"subject": clean_title},
                    )
                )

        return concept_nodes, records
