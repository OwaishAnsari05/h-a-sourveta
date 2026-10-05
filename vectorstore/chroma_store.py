import json,os
EMBEDDING_MODEL="intfloat/multilingual-e5-base"
CHUNKS_PATH="data/chunks.json"
CHROMA_PATH="data/chroma_db"
COLLECTION_NAME="sourveta_multilingual"

def load_chunks(chunks_path=CHUNKS_PATH):
    if not os.path.exists(chunks_path): raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    with open(chunks_path,"r",encoding="utf-8") as f: return json.load(f)

def prepare_chunk_data(chunks):
    ids=[]; documents=[]; metadatas=[]
    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        documents.append(str(chunk.get("text") or ""))
        metadatas.append({"document_id":str(chunk.get("document_id") or ""),"source":str(chunk.get("source") or "unknown"),"page_number":int(chunk["page_number"]) if chunk.get("page_number") is not None else 0,"section":str(chunk.get("section") or "Unknown"),"section_index":int(chunk.get("section_index") or 0),"chunk_index":int(chunk.get("chunk_index") or 0),"language":str(chunk.get("language") or "unknown"),"extraction_method":str(chunk.get("extraction_method") or "unknown"),"ocr_used":bool(chunk.get("ocr_used",False))})
    if not(len(ids)==len(documents)==len(metadatas)): raise ValueError("IDs, documents and metadata counts do not match.")
    return ids,documents,metadatas

def get_chroma_client(chroma_path=CHROMA_PATH):
    import chromadb
    os.makedirs(chroma_path,exist_ok=True)
    return chromadb.PersistentClient(path=chroma_path)

def get_embedding_model(model_name=EMBEDDING_MODEL):
    from vectorstore.embeddings import get_embedding_model as load_embedding_model
    return load_embedding_model(model_name)

def create_collection(client,collection_name=COLLECTION_NAME,reset=False):
    if reset:
        try: client.delete_collection(name=collection_name)
        except Exception: pass
    try: return client.get_collection(name=collection_name)
    except Exception: return client.create_collection(name=collection_name)

def store_chunks(chunks,collection,model):
    from vectorstore.embeddings import encode_passages
    ids,documents,metadatas=prepare_chunk_data(chunks)
    if not documents: return 0
    embeddings=encode_passages(model,documents)
    collection.add(ids=ids,documents=documents,embeddings=embeddings.tolist(),metadatas=metadatas)
    return len(documents)

def rebuild_collection(chunks_path=CHUNKS_PATH,chroma_path=CHROMA_PATH,collection_name=COLLECTION_NAME,model_name=EMBEDDING_MODEL):
    model=get_embedding_model(model_name); chunks=load_chunks(chunks_path); client=get_chroma_client(chroma_path); collection=create_collection(client,collection_name=collection_name,reset=True)
    ids,documents,metadatas=prepare_chunk_data(chunks)
    if not documents: raise ValueError("No chunks available for indexing.")
    from vectorstore.embeddings import encode_passages
    embeddings=encode_passages(model,documents)
    collection.add(ids=ids,documents=documents,embeddings=embeddings.tolist(),metadatas=metadatas)
    if collection.count()!=len(chunks): raise ValueError(f"Count mismatch! Expected {len(chunks)}, but found {collection.count()}.")
    return collection

if __name__=="__main__":
    collection=rebuild_collection()
    print(f"CHROMADB REBUILD COMPLETE | Collection: {COLLECTION_NAME} | Size: {collection.count()}")