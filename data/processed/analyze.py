import json
from collections import defaultdict


def check_is_empty(value):
    """Kiểm tra giá trị có được coi là 'trống' hay không."""
    if value is None:
        return True, "None/null"
    if isinstance(value, str) and value.strip() == "":
        return True, 'Chuỗi rỗng ""'
    if isinstance(value, (list, dict, tuple)) and len(value) == 0:
        return True, "Tập rỗng []/{}"
    return False, None


def analyze_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        print("Lỗi: Dữ liệu trong file JSON không phải là một danh sách (list)!")
        return

    total_chunks = len(chunks)
    print(f" Tổng số chunks đã đọc: {total_chunks}\n")

    if total_chunks == 0:
        print("File JSON rỗng.")
        return

    empty_stats = defaultdict(lambda: {"None": 0, "EmptyStr": 0, "EmptyList": 0, "MissingKey": 0, "TotalEmpty": 0})

    expected_keys = [
        "chunk_id",
        "document_id",
        "document_title",
        "document_type",
        "article_id",
        "article_number",
        "article_title",
        "part_number",
        "part_title",
        "chapter_number",
        "chapter_title",
        "chunk_index",
        "content",
        "embedding_text",
        "source",
        "source_split",
        "path",
    ]

    for chunk in chunks:
        all_keys = set(expected_keys).union(chunk.keys())

        for key in all_keys:
            if key not in chunk:
                empty_stats[key]["MissingKey"] += 1
                empty_stats[key]["TotalEmpty"] += 1
            else:
                val = chunk[key]
                is_empty, empty_type = check_is_empty(val)
                if is_empty:
                    empty_stats[key]["TotalEmpty"] += 1
                    if empty_type == "None/null":
                        empty_stats[key]["None"] += 1
                    elif empty_type == 'Chuỗi rỗng ""':
                        empty_stats[key]["EmptyStr"] += 1
                    elif empty_type == "Tập rỗng []/{}":
                        empty_stats[key]["EmptyList"] += 1

    header = f"{'Tên trường (Field)':<20} | {'Trống / Thiếu':<15} | {'Tỷ lệ %':<10} | {'Chi tiết (None / Rỗng / Khuyết)':<35}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    sorted_fields = sorted(empty_stats.items(), key=lambda x: x[1]["TotalEmpty"], reverse=True)

    for field, counts in sorted_fields:
        total_empty = counts["TotalEmpty"]
        percentage = (total_empty / total_chunks) * 100

        detail_parts = []
        if counts["None"] > 0:
            detail_parts.append(f"None: {counts['None']}")
        if counts["EmptyStr"] > 0:
            detail_parts.append(f'"": {counts["EmptyStr"]}')
        if counts["EmptyList"] > 0:
            detail_parts.append(f"[]: {counts['EmptyList']}")
        if counts["MissingKey"] > 0:
            detail_parts.append(f"Khuyết khóa: {counts['MissingKey']}")

        details = ", ".join(detail_parts) if detail_parts else "Đầy đủ 100%"

        print(f"{field:<20} | {total_empty:<15} | {percentage:>8.2f}% | {details}")

    print("-" * len(header))


if __name__ == "__main__":
    file_path = "chunks.json"
    analyze_chunks(file_path)