from typing import List, Dict, Any
from bson import ObjectId
from app.llm.openai_client import embed_texts
from app.rag.chunk import simple_chunk

# Usa motor no app para obter db (get_db). Aqui, implementamos funções de alto nível.

async def upsert_document(db, title: str, content: str) -> Dict[str, Any]:
    

    # 1) cria documento base em 'documents' e obtém o _id
    res = await db["documents"].insert_one({
        "title": title,
        "content": content,
    })
    document_id = res.inserted_id

    # 2) chunking + embeddings
    chunks = simple_chunk(content)
    embeddings = embed_texts(chunks)

    # 3) remove chunks antigos e insere os novos
    await db["chunks"].delete_many({"document_id": document_id})
    docs = []
    for chunk_text, emb in zip(chunks, embeddings):
        docs.append({
            "document_id": document_id,
            "chunk": chunk_text,
            "embeddings": emb,
        })
    if docs:
        await db["chunks"].insert_many(docs)
    return {"doc_id": str(document_id), "chunks": len(docs)}


async def search_similar_chunks(db, doc_id: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """
    Resolve o documento pelo campo lógico 'doc_id' na coleção 'documents',
    e busca na coleção 'chunks' pelo ObjectId em 'document_id', usando o índice vetorial.
    """
    # doc_id é string de ObjectId
    try:
        oid = ObjectId(doc_id)
    except Exception:
        return []
    doc = await db["documents"].find_one({"_id": oid})
    if not doc:
        return []
    document_id: ObjectId = doc["_id"]

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embeddings",
                "queryVector": query_embedding,
                "numCandidates": 200,
                "limit": k,
                "filter": {"document_id": document_id}
            }
        },
        {"$project": {"_id": 1, "chunk": 1, "document_id": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]
    cursor = db["chunks"].aggregate(pipeline)
    return [d async for d in cursor]