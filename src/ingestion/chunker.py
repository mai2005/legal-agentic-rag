import re
from dataclasses import dataclass, field
from src.ingestion.parser import Article

@dataclass
class Chunk:
    chunk_id: str

    # Bộ luật/Văn bản
    document_id: str
    document_title: str
    document_type: str
    
    # Điều
    article_id: str
    article_number: str
    article_title: str | None

    # Phần
    part_number: str | None
    part_title: str | None

    # Chương
    chapter_number: str | None
    chapter_title: str | None

    chunk_index: int
    content: str
    embedding_text: str

    source: str | None = None
    source_split: str | None = None

    path: list[str] = field(default_factory=list)

class Chunker:
    def __init__(self, max_character: int = 3000, overlap: int = 200) -> None:
        if max_character <= 0:
            raise ValueError("max_character phai lon hon 0")

        if overlap < 0:
            raise ValueError("overlap phai lon hon hoac bang 0")

        if overlap >= max_character:
            raise ValueError("overlap phai nho hon max_character")

        self.max_character = max_character
        self.overlap = overlap

    def chunks(self, articles: list[Article]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for article in articles:
            chunks.extend(self.chunk(article))

        return chunks

    def chunk(self, article: Article) -> list[Chunk]:
        content = article.content.strip()
        
        if not content:
            return []

        if len(content) <= self.max_character:
            return [self._build_chunk(article, chunk_index=0, content=content)]

        content_parts = self._split_long_content(content)

        return [self._build_chunk(article, chunk_index=index, content=part) for index, part in enumerate(content_parts) if part.strip()]

    def _split_long_content(self, content: str) -> list[str]:
        paras = self._split_paras(content)

        chunks: list[str] = []
        current_paras: list[str] = []
        current_length = 0

        for para in paras:
            para = para.strip()

            if not para:
                continue

            if len(para) > self.max_character:
                if current_paras:
                    chunks.append("\n\n".join(current_paras))
                    current_paras = []
                    current_length = 0

                long_parts = self._split_oversized_para(para)
                chunks.extend(long_parts)
                continue

            added_length = len(para)

            if current_paras:
                added_length += 2

            if current_length + added_length > self.max_character and current_paras:
                completed_chunk = "\n\n".join(current_paras)
                chunks.append(completed_chunk)
                overlap_text = self._get_overlap(completed_chunk)
                current_paras = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)

            current_paras.append(para)
            current_length += added_length

        if current_paras:
            chunks.append("\n\n".join(current_paras))

        return self._remove_duplicate_chunks(chunks)

    @staticmethod
    def _split_paras(content: str) -> list[str]:
        paras = re.split(r"\n\s*\n", content.strip())
        
        if len(paras) > 1:
            return paras

        return [line.strip() for line in content.splitlines() if line.strip()]
        
    def _split_oversized_para(self, para: str) -> list[str]:
        words = para.split()

        chunks: list[str] = []
        current_words: list[str] = []
        current_length = 0

        for word in words:
            added_length = len(word)

            if current_words:
                added_length += 1

            if current_length + added_length > self.max_character and current_words:
                completed_chunk = " ".join(current_words)
                chunks.append(completed_chunk)
                overlap_text = self._get_overlap(completed_chunk)
                current_words = overlap_text.split() if overlap_text else []
                current_length = len(" ".join(current_words))

            current_words.append(word)
            current_length += added_length

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks
        
    def _get_overlap(self, content: str) -> str:
        if self.overlap == 0:
            return ""

        if len(content) <= self.overlap:
            return content

        overlap = content[-self.overlap:]

        first_space = overlap.find(" ")
        if first_space != -1:
            overlap = overlap[first_space+1:]

        return overlap.strip()

    def _build_chunk(self, article: Article, chunk_index: int, content: str) -> Chunk:
        chunk_id = f"{article.article_id}-chunk-{chunk_index}"
        embedding_text = self._build_embedding_text(article=article, content=content)
        return Chunk(
            chunk_id=chunk_id,
            document_id=article.document_id,
            document_title=article.document_title,
            document_type=article.document_type,
            article_id=article.article_id,
            article_number=article.article_number,
            article_title=article.article_title,
            part_number=article.part_number,
            part_title=article.part_title,
            chapter_number=article.chapter_number,
            chapter_title=article.chapter_title,
            chunk_index=chunk_index,
            content=content.strip(),
            embedding_text=embedding_text,
            source=article.source,
            source_split=article.source_split,
            path=article.path.copy()
        )

    @staticmethod
    def _build_embedding_text(article: Article, content: str) -> str:
        parts: list[str] = [f"Van ban: {article.document_title}"]
        if article.part_number:
            part_text = f"Phan {article.part_number}"
            if article.part_title:
                part_text += f": {article.part_title}"
            parts.append(part_text)
        if article.chapter_number:
            chapter_text = f"Chuong {article.chapter_number}"
            if article.chapter_title:
                chapter_text += f": {article.chapter_title}"
            parts.append(chapter_text)
        article_heading = f"Dieu {article.article_number}"
        if article.article_title:
            article_heading += f": {article.article_title}"
        parts.append(article_heading)
        parts.append(content.strip())

        return "\n".join(parts)

    @staticmethod
    def _remove_duplicate_chunks(chunks: list[str]) -> list[str]:
        result: list[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if result and chunk == result[-1]:
                continue
            result.append(chunk)
        return result
