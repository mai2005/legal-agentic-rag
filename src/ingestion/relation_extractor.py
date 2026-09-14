import re
from typing import Any
from src.graph.schema import RelationType, RelationshipRecord

class RelationExtractor:

    INTERNAL_ARTICLE_PATTERN = re.compile(
        r"(?:quy\s+định\s+tại|theo|căn\s+cứ|tại)\s+"
        r"(?:(?:điểm\s+[a-zđ0-9]+\s+)?(?:khoản\s+\d+\s+)?)?"
        r"Điều\s+(?P<art_num>\d+[a-zA-Z]?)",
        re.IGNORECASE,
    )

    EXTERNAL_REF_PATTERN = re.compile(
        r"(?:quy\s+định\s+tại|theo|căn\s+cứ|hướng\s+dẫn\s+tại)\s+"
        r"(?:(?:Điều\s+(?P<ext_art>\d+[a-zA-Z]?)\s+)?)?"
        r"(?P<doc_title>(?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Nghị\s+quyết|Quyết\s+định)\s+"
        r"(?:số\s+[0-9]+/[0-9]+/[A-ZĐ0-9\-]+|[A-ZÀ-Ỹa-zà-ỹ0-9\s\-]+?(?=\s*(?:năm|\d{4}|ngày|\.|\,|\;|\n|\)|$))))",
        re.IGNORECASE,
    )

    GUIDES_PATTERN = re.compile(
        r"(?:quy\s+định\s+chi\s+tiết|hướng\s+dẫn\s+thi\s+hành|hướng\s+dẫn\s+thực\s+hiện)\s+"
        r"(?:một\s+số\s+điều\s+của\s+)?(?P<target_doc>(?:Bộ\s+luật|Luật|Nghị\s+định|Pháp\s+lệnh)\s+[A-ZÀ-Ỹa-zà-ỹ0-9\s\-]+?(?=\s*(?:năm|\d{4}|ngày|\.|\,|\;|\n|\)|$)))",
        re.IGNORECASE,
    )

    REPLACES_PATTERN = re.compile(
        r"(?:thay\s+thế|bãi\s+bỏ\s+toàn\s+bộ)\s+"
        r"(?P<target_doc>(?:Nghị\s+định|Thông\s+tư|Quyết\s+định|Luật)\s+(?:số\s+)?[0-9]+/[0-9]+/[A-ZĐ0-9\-]+)",
        re.IGNORECASE,
    )

    AMENDS_PATTERN = re.compile(
        r"(?:sửa\s+đổi[,\s]+bổ\s+sung|bãi\s+bỏ\s+(?:điều|khoản|điểm)\s+\d+.*?của)\s+"
        r"(?P<target_doc>(?:Nghị\s+định|Thông\s+tư|Luật|Quyết\s+định)\s+(?:số\s+)?[0-9]+/[0-9]+/[A-ZĐ0-9\-]+)",
        re.IGNORECASE,
    )

    def __init__(self, linker=None) -> None:
        self.linker = linker

    def extract_internal_article_references(
        self,
        source_doc_id: str,
        source_article_id: str,
        source_article_num: str,
        content: str,
    ) -> list[RelationshipRecord]:
        records: list[RelationshipRecord] = []
        matches = self.INTERNAL_ARTICLE_PATTERN.findall(content)
        seen_targets = set()

        for raw_num in matches:
            target_num = raw_num.strip().lower()
            if target_num == source_article_num.strip().lower():
                continue

            target_art_id = None
            if self.linker:
                target_art_id = self.linker.resolve_article(source_doc_id, target_num)
            
            if not target_art_id:
                target_art_id = f"{source_doc_id}_art_{target_num}" if not self.linker else self.linker.build_canonical_article_id(source_doc_id, target_num)

            if target_art_id not in seen_targets:
                seen_targets.add(target_art_id)
                records.append(
                    RelationshipRecord(
                        from_id=source_article_id,
                        to_id=target_art_id,
                        rel_type=RelationType.REFERENCES,
                        from_label="Article",
                        to_label="Article",
                        properties={
                            "scope": "internal",
                            "target_article_number": target_num,
                        },
                    )
                )

        return records

    def extract_external_references(
        self,
        source_article_id: str,
        content: str,
    ) -> list[RelationshipRecord]:
        records: list[RelationshipRecord] = []
        matches = self.EXTERNAL_REF_PATTERN.finditer(content)
        seen_docs = set()

        for match in matches:
            doc_title = match.group("doc_title").strip()
            ext_art = match.group("ext_art")

            target_doc_id = None
            if self.linker:
                target_doc_id = self.linker.resolve_document(doc_title)

            if target_doc_id and target_doc_id not in seen_docs:
                seen_docs.add(target_doc_id)
                
                if ext_art and self.linker:
                    ext_art_id = self.linker.resolve_article(target_doc_id, ext_art)
                    if ext_art_id:
                        records.append(
                            RelationshipRecord(
                                from_id=source_article_id,
                                to_id=ext_art_id,
                                rel_type=RelationType.REFERENCES,
                                from_label="Article",
                                to_label="Article",
                                properties={
                                    "scope": "external",
                                    "target_doc_title": doc_title,
                                    "target_article_number": ext_art,
                                },
                            )
                        )
                        continue

                records.append(
                    RelationshipRecord(
                        from_id=source_article_id,
                        to_id=target_doc_id,
                        rel_type=RelationType.REFERENCES,
                        from_label="Article",
                        to_label="Document",
                        properties={
                            "scope": "external_document",
                            "raw_text": doc_title,
                        },
                    )
                )

        return records

    def extract_document_evolution_relations(
        self,
        doc_id: str,
        doc_title: str,
        doc_content: str,
    ) -> list[RelationshipRecord]:
        records: list[RelationshipRecord] = []

        guides_matches = self.GUIDES_PATTERN.finditer(doc_content[:3000])  # Thường nằm ở phần đầu
        for m in guides_matches:
            target_text = m.group("target_doc").strip()
            target_id = self.linker.resolve_document(target_text) if self.linker else None
            if target_id and target_id != doc_id:
                records.append(
                    RelationshipRecord(
                        from_id=doc_id,
                        to_id=target_id,
                        rel_type=RelationType.GUIDES,
                        from_label="Document",
                        to_label="Document",
                        properties={"raw_target": target_text},
                    )
                )

        replaces_matches = self.REPLACES_PATTERN.finditer(doc_content[-5000:])  # Thường nằm ở điều khoản thi hành
        for m in replaces_matches:
            target_text = m.group("target_doc").strip()
            target_id = self.linker.resolve_document(target_text) if self.linker else None
            if target_id and target_id != doc_id:
                records.append(
                    RelationshipRecord(
                        from_id=doc_id,
                        to_id=target_id,
                        rel_type=RelationType.REPLACES,
                        from_label="Document",
                        to_label="Document",
                        properties={"raw_target": target_text},
                    )
                )

        amends_matches = self.AMENDS_PATTERN.finditer(doc_content)
        for m in amends_matches:
            target_text = m.group("target_doc").strip()
            target_id = self.linker.resolve_document(target_text) if self.linker else None
            if target_id and target_id != doc_id:
                records.append(
                    RelationshipRecord(
                        from_id=doc_id,
                        to_id=target_id,
                        rel_type=RelationType.AMENDS,
                        from_label="Document",
                        to_label="Document",
                        properties={"raw_target": target_text},
                    )
                )

        return records
