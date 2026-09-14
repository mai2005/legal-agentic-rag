# Tra cứu đa bước các điều khoản dẫn chiếu từ một Điều luật (1 đến 2 hops)
QUERY_MULTI_HOP_REFERENCES = """
MATCH (start:Article {article_id: $article_id})
MATCH path = (start)-[:REFERENCES*1..2]->(target:Article)
MATCH (doc:Document {doc_id: target.doc_id})
OPTIONAL MATCH (ch:Chapter {chapter_id: target.chapter_id})
RETURN DISTINCT
    target.article_id AS article_id,
    target.article_number AS article_number,
    target.article_title AS article_title,
    target.content AS content,
    doc.doc_id AS document_id,
    doc.title AS document_title,
    doc.doc_type AS document_type,
    ch.chapter_number AS chapter_number,
    ch.chapter_title AS chapter_title,
    length(path) AS hop_distance,
    [r in relationships(path) | type(r)] AS relation_path
ORDER BY hop_distance ASC
LIMIT $limit
"""

# Tra cứu các điều khoản tham chiếu đến điều khoản hiện tại
QUERY_INCOMING_REFERENCES = """
MATCH (source:Article)-[:REFERENCES]->(target:Article {article_id: $article_id})
MATCH (doc:Document {doc_id: source.doc_id})
RETURN DISTINCT
    source.article_id AS article_id,
    source.article_number AS article_number,
    source.article_title AS article_title,
    source.content AS content,
    doc.doc_id AS document_id,
    doc.title AS document_title,
    doc.doc_type AS document_type
LIMIT $limit
"""

# Tra cứu văn bản hướng dẫn thi hành
QUERY_GUIDING_DOCUMENTS = """
MATCH (law:Document {doc_id: $doc_id})<-[:GUIDES]-(guide:Document)
OPTIONAL MATCH (guide)-[:CONTAINS]->(art:Article)
RETURN DISTINCT
    guide.doc_id AS guide_doc_id,
    guide.title AS guide_title,
    guide.doc_type AS guide_type,
    art.article_id AS guide_article_id,
    art.article_number AS guide_article_number,
    art.content AS guide_article_content
LIMIT $limit
"""

# Tra cứu văn bản sửa đổi, bổ sung hoặc thay thế
QUERY_LEGAL_EVOLUTION = """
MATCH (target:Document {doc_id: $doc_id})<-[r:AMENDS|REPLACES]-(source:Document)
RETURN DISTINCT
    type(r) AS evolution_type,
    source.doc_id AS doc_id,
    source.title AS title,
    source.doc_type AS doc_type,
    source.doc_number AS doc_number,
    r.scope AS scope,
    r.note AS note
"""

# Tra cứu cấu trúc phân cấp đầy đủ của một Điều
QUERY_ARTICLE_HIERARCHY = """
MATCH (a:Article {article_id: $article_id})
MATCH (d:Document {doc_id: a.doc_id})
OPTIONAL MATCH (p:Part {part_id: a.part_id})
OPTIONAL MATCH (c:Chapter {chapter_id: a.chapter_id})
RETURN
    a.article_id AS article_id,
    a.article_number AS article_number,
    a.article_title AS article_title,
    a.content AS content,
    d.doc_id AS document_id,
    d.title AS document_title,
    d.doc_type AS document_type,
    p.part_number AS part_number,
    p.part_title AS part_title,
    c.chapter_number AS chapter_number,
    c.chapter_title AS chapter_title,
    a.path AS path
"""

# Tìm kiếm điều luật theo từ khóa hoặc mã văn bản + số điều
QUERY_FIND_ARTICLE_BY_NUMBER = """
MATCH (d:Document)
WHERE d.title CONTAINS $doc_query OR d.doc_id = $doc_id
MATCH (d)-[:CONTAINS]->(a:Article)
WHERE a.article_number = $article_number
RETURN
    a.article_id AS article_id,
    a.article_number AS article_number,
    a.article_title AS article_title,
    a.content AS content,
    d.doc_id AS document_id,
    d.title AS document_title,
    d.doc_type AS document_type
LIMIT 5
"""

# Tra cứu Điều luật thông qua Khái niệm / Thuật ngữ pháp lý
QUERY_ARTICLES_BY_CONCEPT = """
MATCH (c:LegalConcept)
WHERE toLower(c.name) CONTAINS toLower($concept_query) OR c.normalized_name CONTAINS $concept_query
MATCH (a:Article)-[r:DEFINES|REGULATES]->(c)
MATCH (d:Document {doc_id: a.doc_id})
RETURN DISTINCT
    c.name AS concept_name,
    type(r) AS relation_type,
    a.article_id AS article_id,
    a.article_number AS article_number,
    a.article_title AS article_title,
    a.content AS content,
    d.doc_id AS document_id,
    d.title AS document_title
LIMIT $limit
"""
