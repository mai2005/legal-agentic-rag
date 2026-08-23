import re
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from src.ingestion.parser import Article


@dataclass
class Chunk:
    chunk_id: str

    # Bộ luật / Văn bản
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
    """
    Token-aware chunker.

    max_tokens được tính trên TOÀN BỘ embedding_text:
        metadata/context + content

    Do đó nếu SPLADE có giới hạn 512 tokens, embedding_text
    được đảm bảo không vượt quá max_tokens.
    """

    def __init__(
        self,
        tokenizer_name: str = "naver/splade-cocondenser-ensembledistil",
        max_tokens: int = 510,
        overlap_tokens: int = 0,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens phai lon hon 0")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens phai lon hon hoac bang 0")

        if overlap_tokens >= max_tokens:
            raise ValueError(
                "overlap_tokens phai nho hon max_tokens"
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

    def chunks(self, articles: list[Article]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for article in articles:
            chunks.extend(self.chunk(article))

        return chunks

    def chunk(self, article: Article) -> list[Chunk]:
        content = article.content.strip()

        if not content:
            return []

        # Kiểm tra toàn bộ embedding_text, không chỉ content.
        if self._embedding_token_length(article, content) <= self.max_tokens:
            return [
                self._build_chunk(
                    article=article,
                    chunk_index=0,
                    content=content,
                )
            ]

        content_parts = self._split_long_content(
            article=article,
            content=content,
        )

        return [
            self._build_chunk(
                article=article,
                chunk_index=index,
                content=part,
            )
            for index, part in enumerate(content_parts)
            if part.strip()
        ]

    def _split_long_content(
        self,
        article: Article,
        content: str,
    ) -> list[str]:
        """
        Split ưu tiên theo paragraph.

        Nếu paragraph làm chunk vượt max_tokens:
            - đóng chunk hiện tại
            - split paragraph theo token-aware word grouping
        """
        paras = self._split_paras(content)

        chunks: list[str] = []
        current_paras: list[str] = []

        for para in paras:
            para = para.strip()

            if not para:
                continue

            # Nếu paragraph tự nó đã quá giới hạn,
            # flush chunk hiện tại trước.
            if not self._fits_chunk(article, para):
                if current_paras:
                    completed_chunk = "\n\n".join(current_paras).strip()
                    if completed_chunk:
                        chunks.append(completed_chunk)

                    current_paras = []

                # Split paragraph lớn.
                long_parts = self._split_oversized_para(
                    article=article,
                    para=para,
                )

                chunks.extend(long_parts)
                continue

            candidate = self._join_paras(current_paras, para)

            if current_paras and not self._fits_chunk(
                article,
                candidate,
            ):
                completed_chunk = "\n\n".join(current_paras).strip()

                if completed_chunk:
                    chunks.append(completed_chunk)

                overlap_text = self._get_overlap(
                    content=completed_chunk,
                )

                current_paras = (
                    [overlap_text]
                    if overlap_text
                    else []
                )

                # Sau overlap, thử thêm paragraph mới.
                candidate = self._join_paras(
                    current_paras,
                    para,
                )

                # Nếu overlap + para vượt giới hạn,
                # bỏ overlap để đảm bảo không vượt max_tokens.
                if current_paras and not self._fits_chunk(
                    article,
                    candidate,
                ):
                    current_paras = [para]
                else:
                    current_paras.append(para)

            else:
                current_paras.append(para)

        if current_paras:
            completed_chunk = "\n\n".join(current_paras).strip()

            if completed_chunk:
                chunks.append(completed_chunk)

        return self._remove_duplicate_chunks(chunks)

    def _split_oversized_para(
        self,
        article: Article,
        para: str,
    ) -> list[str]:
        """
        Split paragraph lớn theo word boundary.

        Mỗi candidate được kiểm tra bằng SPLADE tokenizer
        trên toàn bộ embedding_text.
        """
        words = para.split()

        chunks: list[str] = []
        current_words: list[str] = []

        for word in words:
            candidate_words = current_words + [word]
            candidate = " ".join(candidate_words)

            if current_words and not self._fits_chunk(
                article,
                candidate,
            ):
                completed_chunk = " ".join(current_words).strip()

                if completed_chunk:
                    chunks.append(completed_chunk)

                overlap_text = self._get_overlap(
                    content=completed_chunk,
                )

                current_words = (
                    overlap_text.split()
                    if overlap_text
                    else []
                )

                candidate_words = current_words + [word]
                candidate = " ".join(candidate_words)

                # Nếu overlap khiến chunk vượt giới hạn,
                # bỏ overlap.
                if current_words and not self._fits_chunk(
                    article,
                    candidate,
                ):
                    current_words = [word]
                else:
                    current_words.append(word)

            else:
                current_words.append(word)

        if current_words:
            completed_chunk = " ".join(current_words).strip()

            if completed_chunk:
                chunks.append(completed_chunk)

        return chunks

    def _fits_chunk(
        self,
        article: Article,
        content: str,
    ) -> bool:
        """
        Kiểm tra TOÀN BỘ embedding_text có nằm trong
        giới hạn max_tokens hay không.
        """
        return (
            self._embedding_token_length(article, content)
            <= self.max_tokens
        )

    def _embedding_token_length(
        self,
        article: Article,
        content: str,
    ) -> int:
        embedding_text = self._build_embedding_text(
            article=article,
            content=content,
        )

        encoded = self.tokenizer(
            embedding_text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )

        return len(encoded["input_ids"])

    def _get_overlap(
        self,
        content: str,
    ) -> str:
        """
        Lấy overlap theo token thay vì character.

        Decode phần token cuối thành text.
        """
        if self.overlap_tokens == 0:
            return ""

        encoded = self.tokenizer(
            content,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )

        token_ids = encoded["input_ids"]

        if len(token_ids) <= self.overlap_tokens:
            return content.strip()

        overlap_ids = token_ids[-self.overlap_tokens:]

        overlap_text = self.tokenizer.decode(
            overlap_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return overlap_text.strip()

    @staticmethod
    def _join_paras(
        current_paras: list[str],
        new_para: str,
    ) -> str:
        if not current_paras:
            return new_para

        return "\n\n".join(
            current_paras + [new_para]
        )

    @staticmethod
    def _split_paras(content: str) -> list[str]:
        """
        Ưu tiên split theo paragraph.

        Nếu không có blank line thì split theo từng dòng.
        """
        paras = re.split(
            r"\n\s*\n",
            content.strip(),
        )

        paras = [
            para.strip()
            for para in paras
            if para.strip()
        ]

        if len(paras) > 1:
            return paras

        return [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

    def _build_chunk(
        self,
        article: Article,
        chunk_index: int,
        content: str,
    ) -> Chunk:
        chunk_id = (
            f"{article.article_id}-chunk-{chunk_index}"
        )

        embedding_text = self._build_embedding_text(
            article=article,
            content=content,
        )

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
            path=article.path.copy(),
        )

    @staticmethod
    def _build_embedding_text(
        article: Article,
        content: str,
    ) -> str:
        parts: list[str] = [
            f"Van ban: {article.document_title}"
        ]

        if article.part_number:
            part_text = f"Phan {article.part_number}"

            if article.part_title:
                part_text += f": {article.part_title}"

            parts.append(part_text)

        if article.chapter_number:
            chapter_text = (
                f"Chuong {article.chapter_number}"
            )

            if article.chapter_title:
                chapter_text += (
                    f": {article.chapter_title}"
                )

            parts.append(chapter_text)

        article_heading = (
            f"Dieu {article.article_number}"
        )

        if article.article_title:
            article_heading += (
                f": {article.article_title}"
            )

        parts.append(article_heading)
        parts.append(content.strip())

        return "\n".join(parts)

    @staticmethod
    def _remove_duplicate_chunks(
        chunks: list[str],
    ) -> list[str]:
        result: list[str] = []

        for chunk in chunks:
            chunk = chunk.strip()

            if not chunk:
                continue

            if result and chunk == result[-1]:
                continue

            result.append(chunk)

        return result