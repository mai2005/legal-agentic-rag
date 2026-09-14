# Legal Agentic RAG

An **Agentic Retrieval-Augmented Generation (Agentic RAG)** system tailored for legal information retrieval, analysis, and question answering on Vietnamese legal documents.

The system adopts a state-of-the-art **Multi-Agent Orchestration Workflow** built with **LangGraph**, combining **Hybrid Dense + Sparse Vector Retrieval (BGE-M3)** on **Qdrant**, **Legal Knowledge Graph (Neo4j)**, multi-stage **RRF Fusion & Cross-Encoder Reranking**, and automated **Self-Correction & Anti-Hallucination Verification**.

---

## 🌟 Key Features

- **Query Decomposition & DAG Dependency Planning**: Automatically analyzes complex legal queries, extracts entities, breaks them down into independent sub-queries, and constructs a Directed Acyclic Graph (DAG) for dependency-aware resolution.
- **Context-Aware Query Rewriting**: Optimizes sub-queries with legal terminology and context for targeted retrieval.
- **Hybrid Vector Retrieval**: Combines Dense Embedding (**BGE-M3**) with Sparse/Lexical BM25 & SPLADE indexing in **Qdrant Vector DB**.
- **Legal Knowledge Graph Retrieval**: Navigates hierarchical structures (Document → Part → Chapter → Article) and entity relationships (references, amendments, legal concepts) on **Neo4j**.
- **Multi-Stage Ranking & Fusion**: Merges multi-source results with **Reciprocal Rank Fusion (RRF)** and applies fine-grained semantic scoring using a **BGE Cross-Encoder Reranker**.
- **Evidence Validation & Self-Correction Loop**: Validates the relevance and sufficiency of retrieved legal articles, automatically triggering query rewriting and retry cycles when confidence thresholds are not met.
- **Answer Generation & Hallucination Verification**: Drafts grounded answers citing specific Articles and Legal Decrees, followed by automated verification to eliminate hallucinations.
- **Dual Interfaces (Web API & CLI)**: Built-in production-ready **FastAPI Server** (`/api/query`, `/health`, Swagger UI docs) and command-line execution script.

---

## 🔄 Agentic Workflow Architecture (LangGraph)

```text
               User Legal Query
                      │
                      ▼
               [ Query Analyzer ]
       (Intent classification, NER)
                      │
                      ▼
             [ Query Decomposer ]
       (Split into sub-queries)
                      │
                      ▼
            [ Dependency Planner ]
          (Build DAG dependencies)
                      │
                      ▼
              [ Query Rewriter ] ◄───────────────┐
       (Optimize query keywords)                 │
                      │                          │
                      ▼                          │
             [ Retrieval Router ]                │
                      │                          │
        ┌─────────────┴─────────────┐            │
        ▼                           ▼            │
 [ Vector Retrieval ]      [ Graph Retrieval ]   │ (Self-Correction Loop)
   (Dense + BM25)              (Neo4j Graph)     │
        │                           │            │
        └─────────────┬─────────────┘            │
                      ▼                          │
              [ Fusion & Deduplication ]         │
                      │                          │
                      ▼                          │
                 [ Reranker ]                    │
              (BGE Cross-Encoder)                │
                      │                          │
                      ▼                          │
            [ Evidence Validator ]               │
                      │                          │
               (Evidence valid?)                 │
             ├── Insufficient (Retry) ───────────┤
             └── Valid                           │
                      │                          │
                      ▼                          │
                  [ Answerer ]                   │
             (Draft legal response)              │
                      │                          │
                      ▼                          │
                  [ Verifier ]                   │
           (Hallucination inspection)            │
             ├── Issues found (Retry) ───────────┘
             └── Verified
                      │
                      ▼
                 [ Finalizer ]
                      │
                      ▼
                Final Response
```

---

## 🛠️ Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Agentic Framework** | LangGraph |
| **API Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Embedding Model** | BAAI/bge-m3 (Dense + Sparse BM25 / SPLADE) |
| **Vector Database** | Qdrant |
| **Graph Database** | Neo4j (Cypher, APOC) |
| **Reranker** | BAAI/bge-reranker-v2-m3 (Cross-Encoder) |
| **Inference Backend** | Text Embeddings Inference (TEI) |
| **LLM Engine** | Qwen 2.5, OpenAI-compatible APIs, Google Gemini, Ollama |

---

## 🔮 Future Work

- [ ] **Standardized Benchmark Evaluation**: Triển khai bộ đánh giá tự động (Hit Rate@K, MRR@K, NDCG@K).
- [ ] **Multi-Document Reasoning Enhancement**: Tối ưu hóa suy luận nâng cao liên điều khoản và đối chiếu văn bản hết hiệu lực.