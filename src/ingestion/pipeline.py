import logging
from dataclasses import asdict
from typing import Any

from src.ingestion.loader import RawDocument
from src.ingestion.normalizer import Document, normalize_document
from src.ingestion.parser import DocumentParser, Article
from src.ingestion.chunker import Chunker, Chunk
from src.ingestion.relation_extractor import RelationExtractor
from src.ingestion.entity_extractor import LegalConceptExtractor
from src.graph.entity_linker import EntityLinker
from src.graph.schema import (
    DocumentNode,
    PartNode,
    ChapterNode,
    ArticleNode,
    LegalConceptNode,
    RelationshipRecord,
    RelationType,
)

logger = logging.getLogger("ingestion_pipeline")

class IngestionPipeline:

    def __init__(self, linker: EntityLinker | None = None) -> None:
        self.parser = DocumentParser()
        self.chunker = Chunker()
        self.linker = linker or EntityLinker()
        self.relation_extractor = RelationExtractor(linker=self.linker)
        self.concept_extractor = LegalConceptExtractor()

    def process_raw_documents(
        self,
        raw_documents: list[RawDocument],
        build_chunks: bool = True,
    ) -> dict[str, Any]:
        documents: list[Document] = []
        doc_nodes: list[DocumentNode] = []
        part_nodes: dict[str, PartNode] = {}
        chapter_nodes: dict[str, ChapterNode] = {}
        article_nodes: list[ArticleNode] = []
        concept_nodes_dict: dict[str, LegalConceptNode] = {}
        
        relationships: list[RelationshipRecord] = []
        all_chunks: list[Chunk] = []

        logger.info("Đang phân tích cấu trúc và đăng ký thực thể vào EntityLinker...")
        parsed_doc_articles: list[tuple[Document, list[Article]]] = []

        for raw_doc in raw_documents:
            doc = normalize_document(raw_doc)
            documents.append(doc)
            articles = self.parser.parse(doc)
            parsed_doc_articles.append((doc, articles))

            self.linker.register_document(
                doc_id=doc.id,
                title=doc.title,
                doc_type=doc.type,
                doc_number=None,
            )

            for art in articles:
                self.linker.register_article(
                    doc_id=doc.id,
                    article_number=art.article_number,
                    article_id=art.article_id,
                )

        logger.info("Đang xây dựng cây phân cấp (Hierarchy) và trích xuất quan hệ...")
        for doc, articles in parsed_doc_articles:
            doc_nodes.append(
                DocumentNode(
                    doc_id=doc.id,
                    title=doc.title,
                    doc_type=doc.type,
                    link=doc.link,
                    source=doc.source,
                    source_split=doc.split,
                    status="active",
                )
            )

            doc_evo_rels = self.relation_extractor.extract_document_evolution_relations(
                doc_id=doc.id,
                doc_title=doc.title,
                doc_content=doc.content,
            )
            relationships.extend(doc_evo_rels)

            if build_chunks:
                chunks = self.chunker.chunks(articles)
                all_chunks.extend(chunks)

            for art in articles:
                part_id = None
                if art.part_number:
                    part_id = f"{doc.id}_part_{self.linker.slugify(art.part_number)}"
                    if part_id not in part_nodes:
                        part_nodes[part_id] = PartNode(
                            part_id=part_id,
                            doc_id=doc.id,
                            part_number=art.part_number,
                            part_title=art.part_title,
                        )
                        relationships.append(
                            RelationshipRecord(
                                from_id=doc.id,
                                to_id=part_id,
                                rel_type=RelationType.CONTAINS,
                                from_label="Document",
                                to_label="Part",
                            )
                        )

                chapter_id = None
                if art.chapter_number:
                    chapter_id = f"{doc.id}_chap_{self.linker.slugify(art.chapter_number)}"
                    if chapter_id not in chapter_nodes:
                        chapter_nodes[chapter_id] = ChapterNode(
                            chapter_id=chapter_id,
                            doc_id=doc.id,
                            part_id=part_id,
                            chapter_number=art.chapter_number,
                            chapter_title=art.chapter_title,
                        )
                        parent_id = part_id or doc.id
                        parent_label = "Part" if part_id else "Document"
                        relationships.append(
                            RelationshipRecord(
                                from_id=parent_id,
                                to_id=chapter_id,
                                rel_type=RelationType.CONTAINS,
                                from_label=parent_label,
                                to_label="Chapter",
                            )
                        )

                article_nodes.append(
                    ArticleNode(
                        article_id=art.article_id,
                        doc_id=doc.id,
                        article_number=art.article_number,
                        article_title=art.article_title,
                        content=art.content,
                        part_id=part_id,
                        chapter_id=chapter_id,
                        part_number=art.part_number,
                        chapter_number=art.chapter_number,
                        path=art.path,
                    )
                )

                parent_id = chapter_id or part_id or doc.id
                parent_label = "Chapter" if chapter_id else ("Part" if part_id else "Document")
                relationships.append(
                    RelationshipRecord(
                        from_id=parent_id,
                        to_id=art.article_id,
                        rel_type=RelationType.CONTAINS,
                        from_label=parent_label,
                        to_label="Article",
                    )
                )

                int_refs = self.relation_extractor.extract_internal_article_references(
                    source_doc_id=doc.id,
                    source_article_id=art.article_id,
                    source_article_num=art.article_number,
                    content=art.content,
                )
                relationships.extend(int_refs)

                ext_refs = self.relation_extractor.extract_external_references(
                    source_article_id=art.article_id,
                    content=art.content,
                )
                relationships.extend(ext_refs)

                concepts, concept_rels = self.concept_extractor.extract_from_article(
                    article_id=art.article_id,
                    article_title=art.article_title,
                    article_content=art.content,
                )
                for c in concepts:
                    concept_nodes_dict[c.concept_id] = c
                relationships.extend(concept_rels)

        return {
            "documents": doc_nodes,
            "parts": list(part_nodes.values()),
            "chapters": list(chapter_nodes.values()),
            "articles": article_nodes,
            "concepts": list(concept_nodes_dict.values()),
            "relationships": relationships,
            "chunks": all_chunks,
        }
