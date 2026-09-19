import os
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import create_section_chunks,save_chunks,get_document_chunks_path
from vectorstore.chroma_store import get_chroma_client,get_embedding_model,create_collection,prepare_chunk_data
from api.services.status import update_status

CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "tata_annual_report"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def process_document(pdf_path,document_id,source=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if source is None:
        source = os.path.basename(pdf_path)

    try:
        print(f"Processing document: {source}")

        pages = parse_pdf(pdf_path,document_id=document_id,source=source)
        print(f"Parsed {len(pages)} pages.")

        chunks = create_section_chunks(pages)
        print(f"Created {len(chunks)} chunks.")

        output_path = get_document_chunks_path(document_id)
        save_chunks(chunks,output_path)

        model = get_embedding_model(EMBEDDING_MODEL)
        client = get_chroma_client(CHROMA_PATH)
        collection = create_collection(client,COLLECTION_NAME,reset=False)

        ids,documents,metadatas = prepare_chunk_data(chunks)

        if not documents:
            raise ValueError("No chunks were generated from the document.")

        embeddings = model.encode(documents,show_progress_bar=True)

        existing = collection.get(ids=ids)
        existing_ids = set(existing.get("ids",[]))
        new_items = [
            (i,d,m,e)
            for i,d,m,e in zip(ids,documents,metadatas,embeddings.tolist())
            if i not in existing_ids
        ]

        if new_items:
            new_ids,new_documents,new_metadatas,new_embeddings = zip(*new_items)
            collection.add(
                ids=list(new_ids),
                documents=list(new_documents),
                embeddings=list(new_embeddings),
                metadatas=list(new_metadatas),
            )

        update_status(
            document_id,
            "ready",
            page_count=len(pages),
            chunk_count=len(chunks),
        )

        return {
            "document_id":document_id,
            "filename":source,
            "page_count":len(pages),
            "chunk_count":len(chunks),
            "indexed_chunks":len(new_items),
            "status":"ready",
        }

    except Exception:
        update_status(document_id,"failed")
        raise