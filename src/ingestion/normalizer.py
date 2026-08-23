from dataclasses import dataclass
import re
from src.ingestion.loader import RawDocument

@dataclass
class Document:
    id: str
    title: str
    type: str
    filename: str
    content: str
    length: int
    source: str
    split: str
    link: str | None = None

    @property
    def name(self) -> str:
        return self.title

    @property
    def passage(self) -> str:
        return self.content

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]

    normalize_lines: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalize_lines.append(line)
        previous_blank = is_blank
    
    return "\n".join(normalize_lines).strip()

def normalize_document(raw_document: RawDocument) -> Document:
    content = normalize_text(raw_document.passage)
    
    title = raw_document.name
    link = raw_document.link
    
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

    if not title and link:
        filename = link.split("/")[-1].split(".")[0]
        match = re.match(r"^([A-Z0-9\-]+)", filename)
        if match:
            title = match.group(1)
        else:
            title = filename.replace("-", " ")
            
    if not title:
        title = f"Văn bản {raw_document.id}"

    return Document(
        id=str(raw_document.id).strip(),
        title=title,
        type=doc_type,
        filename=f"context_{raw_document.id}.json",
        content=content,
        length=len(content),
        source=raw_document.source,
        split=raw_document.split,
        link=link
    )