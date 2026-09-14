from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class RelationType(str, Enum):
    # Hierarchy
    CONTAINS = "CONTAINS"
    
    # Cross-references / Citations
    REFERENCES = "REFERENCES"
    
    # Legal Evolution & Validity
    AMENDS = "AMENDS"      
    REPLACES = "REPLACES"    
    GUIDES = "GUIDES"        
    
    # Semantic Concepts
    DEFINES = "DEFINES"      
    REGULATES = "REGULATES"  

@dataclass
class DocumentNode:
    doc_id: str
    title: str
    doc_type: str
    doc_number: str | None = None
    link: str | None = None
    source: str | None = None
    source_split: str | None = None
    status: str = "active"  # active, amended, replaced, expired

@dataclass
class PartNode:
    part_id: str
    doc_id: str
    part_number: str
    part_title: str | None = None

@dataclass
class ChapterNode:
    chapter_id: str
    doc_id: str
    part_id: str | None
    chapter_number: str
    chapter_title: str | None = None

@dataclass
class ArticleNode:
    article_id: str
    doc_id: str
    article_number: str
    article_title: str | None
    content: str
    part_id: str | None = None
    chapter_id: str | None = None
    part_number: str | None = None
    chapter_number: str | None = None
    path: list[str] = field(default_factory=list)

@dataclass
class LegalConceptNode:
    concept_id: str
    name: str
    normalized_name: str
    description: str | None = None
    category: str | None = None

@dataclass
class RelationshipRecord:
    from_id: str
    to_id: str
    rel_type: RelationType | str
    from_label: str
    to_label: str
    properties: dict[str, Any] = field(default_factory=dict)
