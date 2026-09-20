import json
import os

EMBEDDING_MODEL="all-MiniLM-L6-v2"
CHUNKS_PATH="data/chunks.json"
CHROMA_PATH="data/chroma_db"
COLLECTION_NAME="tata_annual_report"

def load_chunks(chunks_path=CHUNKS_PATH):
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with open(chunks_path,"r",encoding="utf-8") as f:
        chunks=json.load(f)
    return chunks

def prepare_chunk_data(chunks):
    ids=[]
    documents=[]
    metadatas=[]

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        documents.append(str(chunk.get("text") or ""))
        metadatas.append({
            "document_id":str(chunk.get("document_id") or ""),
            "source":str(chunk.get("source") or "unknown"),
            "page_number":int(chunk["page_number"]) if chunk.get("page_number") is not None else 0,
            "section":str(chunk.get("section") or "Unknown"),
            "section_index":int(chunk.get("section_index") or 0),
            "chunk_index":int(chunk.get("chunk_index") or 0),
        })

    if not (len(ids)==len(documents)==len(metadatas)):
        raise ValueError("IDs, documents and metadata counts do not match.")

    return ids,documents,metadatas

def get_chroma_client(chroma_path=CHROMA_PATH):
    import chromadb
    os.makedirs(chroma_path,exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)

def get_embedding_model(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

def create_collection(client,collection_name=COLLECTION_NAME,reset=False):
    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.create_collection(name=collection_name)

def store_chunks(chunks,collection,model):
    ids,documents,metadatas=prepare_chunk_data(chunks)

    if not documents:
        return 0

    embeddings=model.encode(documents,show_progress_bar=True)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return len(documents)

def rebuild_collection(
    chunks_path=CHUNKS_PATH,
    chroma_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    model_name=EMBEDDING_MODEL,
):
    print("="*70+"\nLOADING EMBEDDING MODEL\n"+"="*70)
    model=get_embedding_model(model_name)
    print("Embedding model loaded successfully.")

    print("\n"+"="*70+"\nLOADING SECTION-AWARE CHUNKS\n"+"="*70)
    chunks=load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks.")

    print("\n"+"="*70+"\nCONNECTING TO CHROMADB\n"+"="*70)
    client=get_chroma_client(chroma_path)
    print("ChromaDB initialized.")

    print("\n"+"="*70+"\nRESETTING COLLECTION\n"+"="*70)
    collection=create_collection(
        client,
        collection_name=collection_name,
        reset=True,
    )
    print(f"Created collection: {collection_name}")

    print("\n"+"="*70+"\nPREPARING & VALIDATING CHUNKS\n"+"="*70)
    ids,documents,metadatas=prepare_chunk_data(chunks)

    print(
        f"IDs: {len(ids)} | "
        f"Documents: {len(documents)} | "
        f"Metadata: {len(metadatas)}"
    )

    print("\n"+"="*70+"\nCREATING & STORING EMBEDDINGS\n"+"="*70)

    embeddings=model.encode(
        documents,
        show_progress_bar=True,
    )

    print(
        f"Created {len(embeddings)} embeddings "
        f"(Dimension: {len(embeddings[0])})."
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("All chunks stored successfully.")

    count=collection.count()

    if count!=len(chunks):
        raise ValueError(
            f"Count mismatch! Expected {len(chunks)}, but found {count}."
        )

    print("\n"+"="*70+"\nCHROMADB VERIFICATION\n"+"="*70)
    print(f"Total documents in ChromaDB: {count}")
    print("Document count verification: PASS")

    return collection

if __name__=="__main__":
    collection=rebuild_collection()

    print("\n"+"="*70)
    print("CHROMADB REBUILD COMPLETE")
    print("="*70)
    print(
        f"Final collection size: {collection.count()} | "
        f"Collection: {COLLECTION_NAME} | "
        f"Database: {CHROMA_PATH}"
    )
    print("="*70)