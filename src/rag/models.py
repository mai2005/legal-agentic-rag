from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    chunk_id: str

    content: str
    score: float

    document_id: str
    document_title: str | None = None
    document_type: str | None = None

    article_id: str | None = None
    article_number: str | None = None
    article_title: str | None = None

    part_number: str | None = None
    part_title: str | None = None

    chapter_number: str | None = None
    chapter_title: str | None = None

    source: str | None = None
    source_split: str | None = None
