from typing import Any
from src.rag.models import RetrievedChunk

class GraphContextBuilder:
    @staticmethod
    def to_retrieved_chunk(item: dict[str, Any], default_score: float = 0.85) -> RetrievedChunk:
        article_id = item.get("article_id") or item.get("guide_article_id") or item.get("id") or "unknown_node"
        doc_id = item.get("document_id") or item.get("doc_id") or item.get("guide_doc_id") or ""
        doc_title = item.get("document_title") or item.get("title") or item.get("guide_title") or ""
        doc_type = item.get("document_type") or item.get("guide_type") or ""
        
        art_num = item.get("article_number") or item.get("guide_article_number") or ""
        art_title = item.get("article_title") or ""
        content = item.get("content") or item.get("guide_article_content") or ""
        
        score = float(item.get("score", default_score))
        
        hop_distance = item.get("hop_distance")
        if hop_distance and isinstance(hop_distance, (int, float)):
            score = max(0.4, default_score - (hop_distance - 1) * 0.15)

        return RetrievedChunk(
            chunk_id=f"graph_{article_id}",
            content=content,
            score=score,
            document_id=doc_id,
            document_title=doc_title,
            document_type=doc_type,
            article_id=article_id,
            article_number=str(art_num),
            article_title=art_title,
            chapter_number=item.get("chapter_number"),
            chapter_title=item.get("chapter_title"),
            source=item.get("source") or "graph_retrieval",
            source_split=item.get("source_split") or "knowledge_graph",
        )

    @staticmethod
    def format_graph_citations(citations: list[dict[str, Any]]) -> str:
        if not citations:
            return ""

        lines = ["[Quan hệ dẫn chiếu pháp lý liên quan từ Knowledge Graph]:"]
        for c in citations:
            src_num = c.get("source_article_number", "")
            target_title = c.get("document_title", "")
            target_num = c.get("article_number", "")
            hop = c.get("hop_distance", 1)
            lines.append(f"- Điều {src_num} dẫn chiếu tới Điều {target_num} ({target_title}) [khoảng cách {hop} hop]")
        
        return "\n".join(lines)

    @staticmethod
    def format_evolution_warnings(evolutions: list[dict[str, Any]]) -> str:
        if not evolutions:
            return ""

        lines = ["[CẢNH BÁO HIỆU LỰC & SỬA ĐỔI VĂN BẢN]:"]
        for ev in evolutions:
            ev_type = ev.get("evolution_type", "")
            title = ev.get("title", "")
            scope = ev.get("scope") or "toàn bộ hoặc một phần"
            if ev_type == "AMENDS":
                lines.append(f"Văn bản này được SỬA ĐỔI/BỔ SUNG bởi: {title} (Phạm vi: {scope})")
            elif ev_type == "REPLACES":
                lines.append(f"Văn bản này đã bị THAY THẾ HOÀN TOÀN bởi: {title}")
            elif ev_type == "GUIDES":
                lines.append(f"Văn bản được HƯỚNG DẪN THI HÀNH bởi: {title}")
        
        return "\n".join(lines)
