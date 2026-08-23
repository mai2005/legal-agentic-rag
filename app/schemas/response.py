from typing import Any
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    chunk_id: str | None = Field(default=None, description="ID của chunk dữ liệu")
    doc_id: str | None = Field(default=None, description="ID của văn bản luật")
    document_title: str | None = Field(default=None, description="Tên văn bản quy phạm pháp luật")
    article_number: str | None = Field(default=None, description="Số hiệu Điều")
    article_title: str | None = Field(default=None, description="Tiêu đề Điều")
    content: str = Field(..., description="Nội dung điều khoản pháp lý")
    score: float = Field(default=0.0, description="Điểm relevance từ reranker")
    source: str | None = Field(default="vector", description="Nguồn trích xuất (vector/graph/reranker)")

class SubQueryItem(BaseModel):
    id: str = Field(..., description="ID câu hỏi con (sub_q1, sub_q2...)")
    query: str = Field(..., description="Nội dung câu hỏi con")
    purpose: str | None = Field(default=None, description="Mục đích của câu hỏi con")
    dependencies: list[str] = Field(default_factory=list, description="Danh sách ID phụ thuộc")
    rewritten_query: str | None = Field(default=None, description="Truy vấn đã được viết lại tối ưu RAG")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Câu hỏi gốc của người dùng")
    query_type: str | None = Field(default="semantic", description="Phân loại câu hỏi (semantic, legal_relation, legal_lookup)")
    query_entities: list[str] = Field(default_factory=list, description="Thực thể pháp lý trích xuất được")
    final_answer: str = Field(..., description="Câu trả lời pháp lý hoàn chỉnh")
    verified_answer: str | None = Field(default=None, description="Câu trả lời sau khi qua bước kiểm định")
    verification_passed: bool = Field(default=False, description="Kết quả kiểm định tính chính xác")
    verification_issues: list[str] = Field(default_factory=list, description="Các vấn đề phát hiện nếu kiểm định chưa đạt")
    evidence_valid: bool = Field(default=False, description="Đánh giá tính đầy đủ của bằng chứng")
    evidence_score: float = Field(default=0.0, description="Điểm đánh giá bằng chứng")
    sub_queries: list[SubQueryItem] = Field(default_factory=list, description="Danh sách các câu hỏi con")
    dependency_plan: dict[str, list[str]] = Field(default_factory=dict, description="Kế hoạch phụ thuộc giữa các câu hỏi con")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Các căn cứ pháp lý được sử dụng")
    retry_count: int = Field(default=0, description="Số lần retry đã thực hiện")
    execution_trace: list[dict[str, Any]] = Field(default_factory=list, description="Nhật ký luồng thực thi các node trong Agent")

class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    services: dict[str, Any] = Field(default_factory=dict)
