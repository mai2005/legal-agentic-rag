from src.ingestion.loader import Loader

def main() -> None:
    loader = Loader(
        dataset_path="data/selected-contexts",
        source_name="vietnamese-legal-docs",
    )

    print("Đang thử tải các tài liệu thô từ selected-contexts...")
    documents = loader.load_documents(limit=5)
    print(f"Đã tải {len(documents)} tài liệu thành công.")

    for index, doc in enumerate(documents):
        print("\n" + "=" * 60)
        print(f"Document Index: {doc.row_index}")
        print(f"Source: {doc.source}")
        print(f"Split: {doc.split}")
        print(f"ID: {doc.id}")
        print(f"Name: {doc.name}")
        print(f"Link: {doc.link}")

        print("--------------------------------------------------------------------------")
        
        passage_preview = doc.passage.strip() if doc.passage else ""
        if len(passage_preview) > 300:
            passage_preview = passage_preview[:300] + "..."
        print(f"\n{passage_preview}")

if __name__ == "__main__":
    main()