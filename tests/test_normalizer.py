from src.ingestion.loader import Loader
from src.ingestion.normalizer import normalize_document

def main() -> None:
    loader = Loader(
        dataset_path="data/selected-contexts",
        source_name="vietnamese-legal-docs",
    )
    raw_document = loader.get_sample(split="train", index=0)
    document = normalize_document(raw_document)
    print("id: ", document.id)
    print("title: ", document.title)
    print("type: ", document.type)
    print("filename: ", document.filename)
    print("length: ", document.length)
    print("\nPreview:")
    print(document.content[:1000])

if __name__ == "__main__":
    main()
