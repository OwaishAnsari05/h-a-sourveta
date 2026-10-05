import os
from ingestion.pipeline import ingest_pdf
from vectorstore.chroma_store import get_chroma_client,create_collection,prepare_chunk_data
from vectorstore.embeddings import get_embedding_model,encode_passages
from api.services.status import update_status

CHROMA_PATH="data/chroma_db"
COLLECTION_NAME="sourveta_multilingual"
EMBEDDING_MODEL="intfloat/multilingual-e5-base"

def process_document(pdf_path,document_id,source=None):
    if not os.path.exists(pdf_path): raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if source is None: source=os.path.basename(pdf_path)
    try:
        print(f"Processing document: {source}",flush=True)
        result=ingest_pdf(pdf_path,document_id=document_id,source=source)
        pages=result["pages"]; chunks=result["chunks"]
        print(f"Parsed {len(pages)} pages. OCR pages: {result['ocr_pages']}",flush=True)
        print(f"Created {len(chunks)} chunks. Languages: {result['language_counts']}",flush=True)
        if not chunks: raise ValueError("No chunks were generated from the document.")
        model=get_embedding_model(EMBEDDING_MODEL)
        client=get_chroma_client(CHROMA_PATH)
        collection=create_collection(client,COLLECTION_NAME,reset=False)
        ids,documents,metadatas=prepare_chunk_data(chunks)
        existing=collection.get(ids=ids)
        existing_ids=set(existing.get("ids",[]))
        pending=[(i,d,m) for i,d,m in zip(ids,documents,metadatas) if i not in existing_ids]
        if pending:
            new_ids,new_documents,new_metadatas=zip(*pending)
            embeddings=encode_passages(model,list(new_documents))
            collection.add(ids=list(new_ids),documents=list(new_documents),embeddings=embeddings.tolist(),metadatas=list(new_metadatas))
        update_status(document_id,"ready",page_count=len(pages),chunk_count=len(chunks))
        return {"document_id":document_id,"filename":source,"page_count":len(pages),"chunk_count":len(chunks),"indexed_chunks":len(pending),"ocr_pages":result["ocr_pages"],"language_counts":result["language_counts"],"status":"ready"}
    except Exception as exc:
        update_status(document_id,"failed",error=str(exc))
        raise
