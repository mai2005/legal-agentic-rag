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

    print("=" * 80)
    print("DOCUMENT")
    print("=" * 80)
    print("ID:", document.id)
    print("Title:", document.title)
    print("Content length:", document.length)
    print("Number of articles:", len(articles))

    print("\n" + "=" * 80)
    print("FIRST ARTICLES")
    print("=" * 80)

    for article in articles[:5]:
        print("\nArticle ID:", article.article_id)
        print("Path:", " > ".join(article.path))
        print("Part:", article.part_number, "-", article.part_title)
        print(
            "Chapter:",
            article.chapter_number,
            "-",
            article.chapter_title,
        )
        print(
            "Article:",
            article.article_number,
            "-",
            article.article_title,
        )
        print("Content preview:")
        print(article.content[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()