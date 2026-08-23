from src.ingestion.chunker import Chunker
from src.ingestion.loader import Loader
from src.ingestion.normalizer import normalize_document
from src.ingestion.parser import DocumentParser


def main() -> None:
    loader = Loader(
        dataset_path="data/selected-contexts",
        source_name="vietnamese-legal-docs",
    )

    raw_document = loader.get_sample(
        split="train",
        index=0,
    )

    document = normalize_document(raw_document)

    parser = DocumentParser()
    articles = parser.parse(document)

    chunker = Chunker(
        max_character=4000,
        overlap=400,
    )

    chunks = chunker.chunks(articles)

    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print("Document:", document.title)
    print("Number of articles:", len(articles))
    print("Number of chunks:", len(chunks))

    empty_chunks = [
        chunk
        for chunk in chunks
        if not chunk.content.strip()
    ]

    oversized_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.content) > 4000
    ]

    print("Empty chunks:", len(empty_chunks))
    print("Oversized chunks:", len(oversized_chunks))

    print("\n" + "=" * 80)
    print("FIRST CHUNKS")
    print("=" * 80)

    for chunk in chunks[:1]:
        print()
        print("Chunk ID:", chunk.chunk_id)
        print("Article ID:", chunk.article_id)
        print("Path:", " > ".join(chunk.path))
        print("Chunk index:", chunk.chunk_index)
        print("Content length:", len(chunk.content))

        print("\nEmbedding text:")
        print(chunk.embedding_text[:1000])

        print("-" * 80)


if __name__ == "__main__":
    main()