import re
import unicodedata
from typing import Any

class EntityLinker:
    def __init__(self) -> None:
        self.doc_registry: dict[str, dict[str, Any]] = {}
        self.alias_to_doc_id: dict[str, str] = {}
        self.doc_articles: dict[str, dict[str, str]] = {}

    @staticmethod
    def slugify(value: str) -> str:
        if not value:
            return ""
        val = unicodedata.normalize("NFD", str(value))
        val = "".join(c for c in val if unicodedata.category(c) != "Mn")
        val = val.lower().strip()
        val = re.sub(r"[^a-z0-9]+", "_", val)
        return val.strip("_")

    def register_document(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        doc_number: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        self.doc_registry[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "doc_type": doc_type,
            "doc_number": doc_number,
        }

        norm_title = self.slugify(title)
        if norm_title:
            self.alias_to_doc_id[norm_title] = doc_id

        if doc_number:
            norm_num = self.slugify(doc_number)
            self.alias_to_doc_id[norm_num] = doc_id

        if aliases:
            for alias in aliases:
                norm_a = self.slugify(alias)
                if norm_a:
                    self.alias_to_doc_id[norm_a] = doc_id

        match = re.search(r"((?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Quyết\s+định)[\w\s\-]+(?:\d{4})?)", title, re.IGNORECASE)
        if match:
            extracted_alias = self.slugify(match.group(1))
            self.alias_to_doc_id[extracted_alias] = doc_id

    def register_article(self, doc_id: str, article_number: str, article_id: str) -> None:
        if doc_id not in self.doc_articles:
            self.doc_articles[doc_id] = {}
        clean_num = str(article_number).strip().lower()
        self.doc_articles[doc_id][clean_num] = article_id

    def resolve_document(self, text_reference: str) -> str | None:
        if not text_reference:
            return None
        
        norm = self.slugify(text_reference)
        if norm in self.alias_to_doc_id:
            return self.alias_to_doc_id[norm]

        for alias, doc_id in self.alias_to_doc_id.items():
            if alias in norm or norm in alias:
                if len(alias) >= 5 and len(norm) >= 5:
                    return doc_id

        return None

    def resolve_article(self, doc_id: str, article_number: str) -> str | None:
        clean_num = str(article_number).strip().lower()
        articles_map = self.doc_articles.get(doc_id, {})
        
        if clean_num in articles_map:
            return articles_map[clean_num]
        
        num_only = re.sub(r"[^\d]", "", clean_num)
        if num_only in articles_map:
            return articles_map[num_only]

        return None

    def build_canonical_article_id(self, doc_id: str, article_number: str) -> str:
        clean_doc = self.slugify(doc_id)
        clean_art = self.slugify(str(article_number))
        return f"{clean_doc}_art_{clean_art}"
