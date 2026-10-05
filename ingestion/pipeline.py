import os
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import create_section_chunks,get_document_chunks_path,save_chunks


def ingest_pdf(pdf_path,document_id,source=None,output_path=None,ocr_enabled=True,ocr_min_chars=80,ocr_dpi=None,max_chars=1800,overlap_chars=300):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    pages=parse_pdf(pdf_path,document_id=document_id,source=source,ocr_enabled=ocr_enabled,ocr_min_chars=ocr_min_chars,ocr_dpi=ocr_dpi)
    chunks=create_section_chunks(pages,max_chars=max_chars,overlap_chars=overlap_chars)
    if output_path is None:
        output_path=get_document_chunks_path(document_id)
    save_chunks(chunks,output_path)
    language_counts={}
    ocr_pages=0
    denoised_pages=0
    for page in pages:
        language=page.get("language","unknown")
        language_counts[language]=language_counts.get(language,0)+1
        ocr_pages+=int(page.get("ocr_used",False))
        denoised_pages+=int(page.get("denoised",False))
    return {
        "document_id":document_id,
        "source":source or os.path.basename(pdf_path),
        "pages":pages,
        "chunks":chunks,
        "page_count":len(pages),
        "chunk_count":len(chunks),
        "ocr_pages":ocr_pages,
        "denoised_pages":denoised_pages,
        "language_counts":language_counts,
        "output_path":output_path,
    }


def inspect_ingestion(result):
    print("="*70)
    print("INGESTION SUMMARY")
    print("="*70)
    print(f"Pages: {result['page_count']}")
    print(f"Chunks: {result['chunk_count']}")
    print(f"OCR pages: {result['ocr_pages']}")
    print(f"LLM-denoised pages: {result.get('denoised_pages',0)}")
    print(f"Languages: {result['language_counts']}")
    print(f"Chunks saved: {result['output_path']}")
    for page in result["pages"][:5]:
        print(f"Page {page['page_number']}: lang={page['language']} method={page['extraction_method']} chars={page['text_length']}")
