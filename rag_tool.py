"""
RAG Tool — Semantic search over stadium knowledge base using Chroma + Sentence-Transformers.
Enables intelligent Q&A about stadium facilities, rules, and services.
"""

import os
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional

EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stadium_db.sqlite")

# Global instances for caching
_model = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    """Load sentence transformer model (cached globally)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection():
    """Get or create Chroma collection for stadium knowledge."""
    global _collection
    if _collection is None:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
        _collection = client.get_or_create_collection(
            name="stadium_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def index_stadium_data() -> int:
    """
    Index all stadium data into Chroma vector store.
    
    Returns:
        Number of documents indexed
    """
    collection = get_collection()
    model = get_embedding_model()

    if collection.count() > 0:
        return collection.count()

    if not os.path.exists(DB_PATH):
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    # Index facilities
    cursor.execute("""
        SELECT f.facility_type, f.zone, f.floor_level, f.is_accessible, 
               f.description, s.name as stadium_name
        FROM facilities f
        JOIN stadiums s ON f.stadium_id = s.id
    """)
    for row in cursor.fetchall():
        doc = (
            f"Stadium: {row[5]}. Facility: {row[0]} located in {row[1]}, "
            f"Level {row[2]}. {'Wheelchair accessible.' if row[3] else ''} "
            f"{row[4] or ''}"
        )
        documents.append(doc)
        metadatas.append({
            "type": "facility",
            "stadium": row[5],
            "facility_type": row[0],
            "zone": row[1],
            "accessible": str(bool(row[3]))
        })
        ids.append(f"facility_{doc_id}")
        doc_id += 1

    # Index transport options
    cursor.execute("""
        SELECT t.mode, t.details, t.is_accessible, t.estimated_time_min,
               t.cost_estimate, s.name as stadium_name
        FROM transport t
        JOIN stadiums s ON t.stadium_id = s.id
    """)
    for row in cursor.fetchall():
        doc = (
            f"Stadium: {row[5]}. Transport: {row[0]}. {row[1]}. "
            f"Estimated time: {row[3]} minutes. Cost: {row[4]}. "
            f"{'Accessible.' if row[2] else 'Not wheelchair accessible.'}"
        )
        documents.append(doc)
        metadatas.append({
            "type": "transport",
            "stadium": row[5],
            "mode": row[0],
            "accessible": str(bool(row[2]))
        })
        ids.append(f"transport_{doc_id}")
        doc_id += 1

    # Index announcements
    cursor.execute("""
        SELECT a.announcement_type, a.message, a.language, s.name
        FROM announcements a
        JOIN stadiums s ON a.stadium_id = s.id
        WHERE a.is_active = 1
    """)
    for row in cursor.fetchall():
        doc = f"Stadium: {row[3]}. Announcement ({row[0]}): {row[1]}"
        documents.append(doc)
        metadatas.append({
            "type": "announcement",
            "stadium": row[3],
            "announcement_type": row[0]
        })
        ids.append(f"announcement_{doc_id}")
        doc_id += 1

    conn.close()

    if not documents:
        return 0

    # Batch insert
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = model.encode(batch_docs).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )

    return len(documents)


def semantic_search(query: str, n_results: int = 10,
                    stadium_filter: Optional[str] = None,
                    type_filter: Optional[str] = None) -> list:
    """
    Search stadium knowledge base semantically.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return
        stadium_filter: Optional stadium name filter
        type_filter: Optional document type filter (facility/transport/announcement)
    
    Returns:
        List of matching documents with metadata and similarity scores
    """
    collection = get_collection()
    model = get_embedding_model()

    if collection.count() == 0:
        index_stadium_data()

    if collection.count() == 0:
        return []

    query_embedding = model.encode([query]).tolist()

    where_filter = None
    if stadium_filter and type_filter:
        where_filter = {
            "$and": [
                {"stadium": stadium_filter},
                {"type": type_filter}
            ]
        }
    elif stadium_filter:
        where_filter = {"stadium": stadium_filter}
    elif type_filter:
        where_filter = {"type": type_filter}

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": round(1 - results["distances"][0][i], 4)
                        if results["distances"] else 0
                })

        return output

    except Exception as e:
        return [{"document": f"Search error: {str(e)}", "metadata": {}, "similarity": 0}]


def find_accessible_facilities(stadium_name: str, facility_type: str = "") -> list:
    """
    Find accessible facilities using semantic search.
    
    Args:
        stadium_name: Name of the stadium
        facility_type: Optional specific facility type
    
    Returns:
        List of accessible facilities
    """
    query = f"wheelchair accessible {facility_type} at {stadium_name}"
    results = semantic_search(query, n_results=10, stadium_filter=stadium_name)
    return [r for r in results if r["metadata"].get("accessible") == "True"]
