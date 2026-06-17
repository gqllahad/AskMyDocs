"""
retriever.py — Finds the most relevant chunks for a user's question
using semantic (vector) search against ChromaDB.

The query is embedded with the same model used during ingestion,
then compared against all stored chunks using cosine similarity.
"""

from ingest.embedder import get_collection

TOP_K = 5             
MIN_SIMILARITY = 0.30     


def retrieve(query: str, top_k: int = TOP_K, session_id: str = None) -> list[dict]:
    """
    Search the vector store for chunks relevant to the query.

    Returns a list of result dicts sorted by relevance (best first):
        {
            "text":       str,   # chunk content
            "source":     str,   # filename
            "page":       int,   # page number
            "score":      float, # cosine similarity (higher = better)
        }
    """
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError("Vector store is empty. Run ingest first.")

    # results = collection.query(
    #     query_texts=[query],
    #     n_results=min(top_k, collection.count()),
    #     include=["documents", "metadatas", "distances"],
    # )

    query_kwargs = {
        "query_texts": [query],
        "n_results": min(top_k * 3, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
     
    results = collection.query(**query_kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        
        # Session boundary check
        meta_session = meta.get("session_id")
        if session_id is not None:
            if meta_session is not None and meta_session != session_id:
                continue
    
        similarity = 1 - (dist / 2)

        if similarity < MIN_SIMILARITY:
            continue  

        chunks.append({
            "text":   doc,
            "source": meta["source"],
            "page":   meta["page"],
            "score":  round(similarity, 3),
        })
        
        if len(chunks) >= top_k:
            break

    return chunks
