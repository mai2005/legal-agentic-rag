from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi pháp lý cần tra cứu và giải đáp", min_length=2)
    max_sub_queries: int = Field(default=4, ge=1, le=8, description="Số lượng câu hỏi con tối đa khi phân rã")
    vector_top_k: int = Field(default=20, ge=1, le=100, description="Top-k retrieval cho vector search")
    rerank_top_k: int = Field(default=8, ge=1, le=50, description="Top-k sau reranking")
    min_evidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Điểm relevance tối thiểu của bằng chứng")
    max_retries: int = Field(default=2, ge=0, le=5, description="Số lần retry tối đa khi bằng chứng hoặc thẩm định không đạt")
