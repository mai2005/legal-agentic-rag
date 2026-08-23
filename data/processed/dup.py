import json
from collections import defaultdict


def check_duplicate_chunk_ids(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file JSON: {e}")
        return

    if not isinstance(chunks, list):
        print("Lỗi: Dữ liệu JSON phải là một danh sách (list)!")
        return

    print(f" Tổng số chunks đã tải: {len(chunks)}")

    id_tracker = defaultdict(list)
    missing_id_count = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")

        if chunk_id is None or (isinstance(chunk_id, str) and chunk_id.strip() == ""):
            missing_id_count += 1
        else:
            id_tracker[chunk_id].append(idx)
    duplicates = {cid: indices for cid, indices in id_tracker.items() if len(indices) > 1}

    print("\n" + "=" * 60)
    print(" KẾT QUẢ KIỂM TRA TRÙNG LẶP CHUNK_ID")
    print("=" * 60)

    if missing_id_count > 0:
        print(f"⚠️ Cảnh báo: Có {missing_id_count} chunk bị thiếu/rỗng chunk_id!")

    if not duplicates:
        print(" THÀNH CÔNG: Không phát hiện chunk_id nào bị trùng lặp.")
    else:
        total_dup_ids = len(duplicates)
        total_dup_records = sum(len(indices) for indices in duplicates.values())

        print(
            f"❌ PHÁT HIỆN TRÙNG LẶP: Có {total_dup_ids} mã chunk_id bị trùng (ảnh hưởng đến {total_dup_records} bản ghi).\n"
        )
        print(f"{'STT':<5} | {'chunk_id bị trùng':<38} | {'Số lần':<8} | Các vị trí Index")
        print("-" * 75)

        for i, (cid, indices) in enumerate(duplicates.items(), 1):
            indices_str = ", ".join(map(str, indices[:5]))
            if len(indices) > 5:
                indices_str += f"... (+{len(indices) - 5} vị trí khác)"

            print(f"{i:<5} | {cid:<38} | {len(indices):<8} | [{indices_str}]")

    print("=" * 60)


if __name__ == "__main__":
    file_path = "chunks.json"
    check_duplicate_chunk_ids(file_path)