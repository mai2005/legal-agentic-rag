from dataclasses import dataclass
from typing import Any, Iterator
from pathlib import Path
import json

@dataclass
class RawDocument:
    source: str
    split: str
    row_index: int
    id: int
    link: str
    passage: str
    name: str | None = None

class Loader:
    def __init__(self, dataset_path: str | Path, source_name: str="vietnamese-legal-docs") -> None:
        self.dataset_path = Path(dataset_path)
        self.source_name = source_name
        self._documents: list[RawDocument] | None = None

    def _load_all_json_documents(self, split: str="train") -> list[RawDocument]:
        if self._documents is not None:
            return self._documents

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Not found dataset path {self.dataset_path}")

        documents = []
        if self.dataset_path.is_dir():
            json_files = sorted(list(self.dataset_path.glob("*.json")), key=lambda p: p.name)
        elif self.dataset_path.suffix == ".json":
            json_files = [self.dataset_path]
        else:
            json_files = []

        for row_index, file_path in enumerate(json_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                doc_id = int(data["id"])
                link = data.get("link", "")
                passage = data.get("passage", "")
                name = data.get("name")  # Có thể None nếu không tồn tại

                documents.append(
                    RawDocument(
                        source=self.source_name,
                        split=split,
                        row_index=row_index,
                        id=doc_id,
                        link=link,
                        passage=passage,
                        name=name
                    )
                )
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")

        self._documents = documents
        return self._documents

    def get_schema(self, split: str="train") -> dict[str, Any]:
        docs = self._load_all_json_documents(split)
        columns = ["source", "split", "row_index", "id", "link", "passage", "name"]
        return {
            "split": split,
            "num_rows": len(docs),
            "columns": columns,
            "features": "RawDocument dataclass containing id, link, passage, name"
        }

    def load_split(self, split: str="train") -> list[RawDocument]:
        return self._load_all_json_documents(split)

    def iter_documents(self, split: str="train", limit: int | None=None) -> Iterator[RawDocument]:
        docs = self._load_all_json_documents(split)
        total_rows = len(docs)
        if limit is not None:
            total_rows = min(limit, total_rows)
        for i in range(total_rows):
            yield docs[i]
    
    def load_documents(self, split: str="train", limit: int | None=None) -> list[RawDocument]:
        return list(self.iter_documents(split=split, limit=limit))

    def get_sample(self, split: str="train", index: int=0) -> RawDocument:
        docs = self._load_all_json_documents(split)
        if index < 0 or index >= len(docs):
            raise IndexError(f"Index {index} out of range (total {len(docs)})")
        return docs[index]