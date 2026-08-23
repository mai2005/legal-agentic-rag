import os
import json
from pathlib import Path
from collections import defaultdict

def run_eda() -> None:
    context_dir = Path("data/selected-contexts")
    if not context_dir.exists():
        print(f"Lỗi: Thư mục {context_dir.resolve()} không tồn tại.")
        return

    json_files = list(context_dir.glob("*.json"))
    print(f"Tổng số file .json tìm thấy: {len(json_files)}")

    doc_type_counts = defaultdict(int)
    doc_type_samples = defaultdict(list)

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            link = data.get("link", "")
            if not link:
                doc_type = "Không có link"
            else:
                filename = link.split("/")[-1].split(".")[0].lower()
                
                if filename.startswith("qc") or filename.startswith("qqc"):
                    doc_type = "QCVN"
                elif filename.startswith("quyet-dinh") or filename.startswith("quyet_dinh"):
                    doc_type = "Quyet-dinh"
                elif filename.startswith("nghi-dinh") or filename.startswith("nghi_dinh") or filename.startswith("decree"):
                    doc_type = "Nghi-dinh"
                elif filename.startswith("thong-tu") or filename.startswith("thong_tu"):
                    doc_type = "Thong-tu"
                elif filename.startswith("bo-luat") or filename.startswith("bo_luat"):
                    doc_type = "Bo-luat"
                elif filename.startswith("luat"):
                    doc_type = "Luat"
                elif filename.startswith("nghi-quyet") or filename.startswith("nghi_quyet"):
                    doc_type = "Nghi-quyet"
                elif filename.startswith("phap-lenh"):
                    doc_type = "Phap-lenh"
                elif filename.startswith("hien-phap"):
                    doc_type = "Hien-phap"
                elif filename.startswith("chi-thi"):
                    doc_type = "Chi-thi"
                elif filename.startswith("huong-dan"):
                    doc_type = "Huong-dan"
                elif filename.startswith("dieu-le"):
                    doc_type = "Dieu-le"
                elif filename.startswith("cong-uoc"):
                    doc_type = "Cong-uoc"
                elif filename.startswith("thong-bao"):
                    doc_type = "Thong-bao"
                elif filename.startswith("hiep-dinh"):
                    doc_type = "Hiep-dinh"
                else:
                    doc_type = "TCVN"
            
            doc_type_counts[doc_type] += 1
            if len(doc_type_samples[doc_type]) < 3:
                doc_type_samples[doc_type].append(file_path.name)

        except Exception as e:
            print(f"Lỗi xử lý file {file_path.name}: {e}")

    print("\n" + "=" * 60)
    print("KẾT QUẢ PHÂN TÍCH DỮ LIỆU THÔ (EDA)")
    print("=" * 60)
    for doc_type, count in sorted(doc_type_counts.items(), key=lambda x: x[1], reverse=True):
        samples = doc_type_samples[doc_type]
        print(f"\nLoại tài liệu (doc_type): {doc_type}")
        print(f"  - Tổng số lượng file .json: {count}")
        print(f"  - 3 file mẫu: {', '.join(samples)}")

if __name__ == "__main__":
    run_eda()
