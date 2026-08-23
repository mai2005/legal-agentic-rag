import json
from pathlib import Path
from dataclasses import asdict

from src.ingestion.chunker import Chunker
from src.ingestion.loader import Loader
from src.ingestion.normalizer import normalize_document
from src.ingestion.parser import DocumentParser

def main() -> None:
    loader = Loader(
        dataset_path="data/selected-contexts",
        source_name="vietnamese-legal-docs",
    )

    raw_documents = loader.load_documents(split="train")

    parser = DocumentParser()
    
    chunker = Chunker()

    all_chunks = []
    total_articles = 0

    print("Đang xử lý toàn bộ tài liệu...")
    for raw_doc in raw_documents:
        document = normalize_document(raw_doc)
        articles = parser.parse(document)
        chunks = chunker.chunks(articles)

        total_articles += len(articles)
        all_chunks.extend(chunks)

    print("\n" + "=" * 80)
    print("THỐNG KÊ TỔNG THỂ")
    print("=" * 80)
    print("Tổng số tài liệu:", len(raw_documents))
    print("Tổng số điều/bài (articles):", total_articles)
    print("Tổng số chunks:", len(all_chunks))

    empty_chunks = [c for c in all_chunks if not c.content.strip()]
    oversized_chunks = [c for c in all_chunks if len(c.content) > 3000]

    print("Số chunk rỗng:", len(empty_chunks))
    print("Số chunk vượt kích thước (>3000 ký tự):", len(oversized_chunks))

    chunks_data = [asdict(c) for c in all_chunks]

    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu thành công {len(chunks_data)} chunks vào: {output_path.resolve()}")


if __name__ == "__main__":
    main()