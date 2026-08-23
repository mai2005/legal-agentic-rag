from dataclasses import dataclass, field
import re
import unicodedata
from src.ingestion.normalizer import Document

@dataclass
class Article:
    article_id: str
    document_id: str
    document_title: str
    document_type: str

    article_number: str
    article_title: str | None
    content: str

    part_number: str | None = None
    part_title: str | None = None

    chapter_number: str | None = None
    chapter_title: str | None = None

    source: str | None = None
    source_split: str | None = None

    path: list[str] = field(default_factory=list)

@dataclass
class Heading:
    heading_type: str
    number: str | None
    title: str | None

class DocumentParser:
    PREFIX_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+")
    PART_PATTERN = re.compile(
        r"^PHẦN\s+"
        r"(?P<number>(?:THỨ\s+)?(?:[A-ZÀ-Ỹ]+|[IVXLCDM]+|\d+))"
        r"(?:\s*[:.\-–—]\s*(?P<title>.*))?$",
        re.IGNORECASE,
    )
    CHAPTER_PATTERN = re.compile(
        r"^CHƯƠNG\s+"
        r"(?P<number>[IVXLCDM]+|\d+)"
        r"(?:\s*[:.\-–—]\s*(?P<title>.*))?$",
        re.IGNORECASE,
    )
    ARTICLE_PATTERN = re.compile(
        r"^ĐIỀU\s+"
        r"(?P<number>\d+[a-zA-ZÀ-Ỹ0-9]?)"
        r"\s*[.]\s*"
        r"(?P<title>.*)$",
        re.IGNORECASE,
    )
    def parse(self, document: Document) -> list[Article]:
        lines = document.content.splitlines()
        articles: list[Article] = []
        current_part_number: str | None = None
        current_part_title: str | None = None
        current_chapter_number: str | None = None
        current_chapter_title: str | None = None
        current_article_number: str | None = None
        current_article_title: str | None = None
        current_article_lines: list[str] = []

        pending_heading_type: str | None = None

        for raw_line in lines:
            line = self._clean_line(raw_line)

            if not line:
                if current_article_number is not None:
                    current_article_lines.append("")
                continue

            if pending_heading_type == "part":
                current_part_title = line
                pending_heading_type = None
                continue
            if pending_heading_type == "chapter":
                current_chapter_title = line
                pending_heading_type = None
                continue

            heading = self._parse_heading(line)

            if heading is None:
                if current_article_number is not None:
                    current_article_lines.append(line)
                continue
            
            if heading.heading_type == "part":
                if current_article_number is not None:
                    articles.append(
                        self._build_article(
                            document=document,
                            article_number=current_article_number,
                            article_title=current_article_title,
                            article_lines=current_article_lines,
                            part_number=current_part_number,
                            part_title=current_part_title,
                            chapter_number=current_chapter_number,
                            chapter_title=current_chapter_title
                        )
                    )
                    current_article_number = None
                    current_article_title = None
                    current_article_lines = []
                current_part_number = heading.number
                current_part_title = heading.title
                current_chapter_number = None
                current_chapter_title = None
                if heading.title is None:
                    pending_heading_type = "part"
                continue

            if heading.heading_type == "chapter":
                if current_article_number is not None:
                    articles.append(
                        self._build_article(
                            document=document,
                            article_number=current_article_number,
                            article_title=current_article_title,
                            article_lines=current_article_lines,
                            part_number=current_part_number,
                            part_title=current_part_title,
                            chapter_number=current_chapter_number,
                            chapter_title=current_chapter_title,
                        )
                    )
                    current_article_number = None
                    current_article_title = None
                    current_article_lines = []
                current_chapter_number = heading.number
                current_chapter_title = heading.title
                if heading.title is None:
                    pending_heading_type = "chapter"
                continue

            if heading.heading_type == "article":
                if current_article_number is not None:
                    articles.append(
                        self._build_article(
                            document=document,
                            article_number=current_article_number,
                            article_title=current_article_title,
                            article_lines=current_article_lines,
                            part_number=current_part_number,
                            part_title=current_part_title,
                            chapter_number=current_chapter_number,
                            chapter_title=current_chapter_title,
                        )
                    )
                current_article_number = heading.number
                current_article_title = heading.title
                current_article_lines = []
                
        if current_article_number is not None:
            articles.append(
                self._build_article(
                    document=document,
                    article_number=current_article_number,
                    article_title=current_article_title,
                    article_lines=current_article_lines,
                    part_number=current_part_number,
                    part_title=current_part_title,
                    chapter_number=current_chapter_number,
                    chapter_title=current_chapter_title,
                )
            )
        if not articles and document.content.strip():
            articles.append(
                Article(
                    article_id=self._slugify(str(document.id) + "-full"),
                    document_id=document.id,
                    document_title=document.title,
                    document_type=document.type,
                    article_number="1",
                    article_title=document.title,
                    content=document.content,
                    source=document.source,
                    source_split=document.split,
                    path=[document.title]
                )
            )
        
        return articles
            
    def _parse_heading(self, line: str) -> Heading | None:
        patterns = [
            ("part", self.PART_PATTERN),
            ("chapter", self.CHAPTER_PATTERN),
            ("article", self.ARTICLE_PATTERN),
        ]
        for heading_type, pattern in patterns:
            match = pattern.match(line)
            if match is None:
                continue
            number = self._clean_optional_text(match.groupdict().get("number"))
            title = self._clean_optional_text(match.groupdict().get("title"))
            return Heading(heading_type, number, title)
        return None

    def _build_article(
        self, 
        document: Document,
        article_number: str,
        article_title: str | None,
        article_lines: list[str],
        part_number: str | None,
        part_title: str | None,
        chapter_number: str | None,
        chapter_title: str | None,
    ) -> Article:
        content = self._normalize_article_content(article_lines)
        part = part_number if part_number else ""
        chapter = chapter_number if chapter_number else ""
        article_id = self._build_article_id(document.id, part, chapter, article_number)
        path = self._build_path(document.title, part_number, chapter_number, article_number)
        
        return Article(
            article_id=article_id,
            document_id=document.id,
            document_title=document.title,
            document_type=document.type,
            article_number=article_number,
            article_title=article_title,
            content=content,
            part_number=part_number,
            part_title=part_title,
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            source=document.source,
            source_split=document.split,
            path=path
        )

    def _clean_line(self, line: str) -> str:
        line = line.strip()
        line = self.PREFIX_PATTERN.sub("", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("*", "")
        line = line.replace("_", "")

        return line.strip()

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip(" \t\r\n.:;-")
        return value or None
        
    @staticmethod
    def _normalize_article_content(lines: list[str]) -> str:
        normalized_lines: list[str] = []
        previous_blank = False

        for line in lines:
            cleand = line.rstrip()
            is_blank = False
            if is_blank and previous_blank:
                continue
            normalized_lines.append(cleand)
            previous_blank = is_blank

        return "\n".join(normalized_lines).strip()

    def _build_article_id(self, document_id: str, part_number: str, chapter_number: str, article_number: str) -> str:
        safe_doc_id = self._slugify(document_id+"-"+part_number+"-"+chapter_number+"-"+article_number)
        return safe_doc_id

    @staticmethod
    def _build_path(
        document_title: str,
        part_number: str | None,
        chapter_number: str | None,
        article_number: str,
    ) -> list[str]:
        path = [document_title]
        if part_number is not None:
            path.append(f"Phần_{part_number}")
        if chapter_number is not None:
            path.append(f"Chương_{chapter_number}")
        path.append(f"Điều {article_number}")   

        return path

    @staticmethod
    def _slugify(value: str) -> str:
        value = unicodedata.normalize("NFD", value)
        value = "".join(c for c in value if unicodedata.category(c) != "Mn")
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value