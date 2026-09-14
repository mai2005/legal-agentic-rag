import os
import sys
import time
import logging
from dataclasses import asdict
from collections import defaultdict
from dotenv import load_dotenv

from src.ingestion.loader import Loader
from src.ingestion.pipeline import IngestionPipeline
from src.graph.entity_linker import EntityLinker
from src.storage.neo4j_store import Neo4jStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_graph")

def main() -> None:
    load_dotenv()

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/selected-contexts"
    recreate = "--recreate" in sys.argv

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

    print("=" * 80)
    print("PIPELINE XÂY DỰNG LEGAL KNOWLEDGE GRAPH (NEO4J)")
    print("=" * 80)
    print(f"Dataset Path: {dataset_path}")
    print(f"Neo4j URI   : {neo4j_uri}")
    print(f"Neo4j DB    : {neo4j_db}")
    print(f"Recreate DB : {recreate}")
    print("=" * 80)

    # 1. Kết nối Neo4j
    store = Neo4jStore(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_db,
    )

    if not store.verify_connectivity():
        logger.error("Không thể kết nối đến Neo4j! Vui lòng kiểm tra container đang chạy: `docker-compose up -d neo4j`")
        sys.exit(1)

    logger.info("Kết nối Neo4j thành công.")

    if recreate:
        logger.warning("Đang xóa toàn bộ dữ liệu cũ trong Neo4j (--recreate)...")
        store.clear_database()

    # 2. Thiết lập Index & Constraints
    logger.info("Đang tạo Constraints & Indexes...")
    store.ensure_constraints_and_indexes()

    # 3. Đọc dữ liệu từ dataset
    logger.info(f"Đang đọc dữ liệu từ thư mục {dataset_path}...")
    loader = Loader(dataset_path=dataset_path, source_name="vietnamese-legal-docs")
    raw_documents = loader.load_documents(split="train")
    logger.info(f"Đã đọc {len(raw_documents)} tài liệu thô.")

    # 4. Xử lý bóc tách phân cấp, trích xuất thực thể & quan hệ
    start_time = time.time()
    linker = EntityLinker()
    pipeline = IngestionPipeline(linker=linker)

    logger.info("Đang thực thi IngestionPipeline...")
    processed = pipeline.process_raw_documents(raw_documents, build_chunks=False)

    docs_data = [asdict(d) for d in processed["documents"]]
    parts_data = [asdict(p) for p in processed["parts"]]
    chapters_data = [asdict(c) for c in processed["chapters"]]
    articles_data = [asdict(a) for a in processed["articles"]]
    concepts_data = [asdict(lc) for lc in processed["concepts"]]
    relationships = processed["relationships"]

    logger.info(f"Bóc tách xong: {len(docs_data)} Documents, {len(parts_data)} Parts, {len(chapters_data)} Chapters, {len(articles_data)} Articles, {len(concepts_data)} LegalConcepts, {len(relationships)} Relationships.")

    # 5. Batch Upsert Nodes vào Neo4j
    logger.info("Đang nạp Nodes vào Neo4j...")
    store.batch_upsert_nodes("Document", "doc_id", docs_data)
    store.batch_upsert_nodes("Part", "part_id", parts_data)
    store.batch_upsert_nodes("Chapter", "chapter_id", chapters_data)
    store.batch_upsert_nodes("Article", "article_id", articles_data)
    store.batch_upsert_nodes("LegalConcept", "concept_id", concepts_data)

    # 6. Gom nhóm và Batch Upsert Relationships
    logger.info("Đang nạp Relationships vào Neo4j...")
    grouped_rels = defaultdict(list)
    for r in relationships:
        key = (
            str(r.rel_type.value if hasattr(r.rel_type, "value") else r.rel_type),
            r.from_label,
            "doc_id" if r.from_label == "Document" else ("part_id" if r.from_label == "Part" else ("chapter_id" if r.from_label == "Chapter" else "article_id")),
            r.to_label,
            "doc_id" if r.to_label == "Document" else ("part_id" if r.to_label == "Part" else ("chapter_id" if r.to_label == "Chapter" else ("concept_id" if r.to_label == "LegalConcept" else "article_id"))),
        )
        grouped_rels[key].append({
            "from_id": r.from_id,
            "to_id": r.to_id,
            "properties": r.properties,
        })

    for (rel_type, from_label, from_key, to_label, to_key), rel_items in grouped_rels.items():
        logger.info(f"  - Đang nạp {len(rel_items)} quan hệ `[:{rel_type}]` ({from_label} -> {to_label})...")
        store.batch_upsert_relationships(
            rel_type=rel_type,
            from_label=from_label,
            from_key=from_key,
            to_label=to_label,
            to_key=to_key,
            rels=rel_items,
        )

    elapsed = time.time() - start_time

    # 7. Thống kê kết quả
    total_docs = store.count_nodes("Document")
    total_arts = store.count_nodes("Article")
    total_concepts = store.count_nodes("LegalConcept")
    total_all_nodes = store.count_nodes()
    total_all_rels = store.count_relationships()

    print("\n" + "=" * 80)
    print("THỐNG KÊ HOÀN TẤT XÂY DỰNG KNOWLEDGE GRAPH TRÊN NEO4J")
    print("=" * 80)
    print(f"⏱️  Thời gian thực thi   : {elapsed:.2f} giây")
    print(f"📦 Tổng số Node trong DB : {total_all_nodes:,}")
    print(f"   ├─ Document           : {total_docs:,}")
    print(f"   ├─ Article            : {total_arts:,}")
    print(f"   └─ LegalConcept       : {total_concepts:,}")
    print(f"🔗 Tổng số Relationship  : {total_all_rels:,}")
    print("=" * 80)
    print("Xây dựng đồ thị tri thức pháp lý thành công!")

    store.close()

if __name__ == "__main__":
    main()
