import os,re
import pymupdf
from ingestion.language import normalize_unicode
from ingestion.metadata import build_page_metadata
from ingestion.ocr import ocr_available,ocr_page_with_metadata,should_ocr,choose_text
from ingestion.ocr_denoiser import denoise_page

def clean_text(text):
    text=normalize_unicode(text)
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def remove_headers(text,header_candidates):
    if not header_candidates:
        return text
    candidates={normalize_unicode(x).strip() for x in header_candidates}
    return "\n".join(line.strip() for line in text.splitlines() if line.strip() not in candidates)

def inspect_pages(pages,page_numbers):
    for page_number in page_numbers:
        if page_number<1 or page_number>len(pages):
            continue
        page=pages[page_number-1]
        print(f"\n===== PAGE {page_number} =====")
        print(f"LANGUAGE: {page.get('language')} | METHOD: {page.get('extraction_method')} | OCR: {page.get('ocr_used')} | OCR_SCORE: {page.get('ocr_quality_score')}")
        print(page["text"][:1000])

def parse_pdf(pdf_path,document_id,source=None,header_candidates=None,ocr_enabled=True,ocr_min_chars=80,ocr_dpi=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    header_candidates=set() if header_candidates is None else header_candidates
    source=source or os.path.basename(pdf_path)
    pages=[]
    ocr_ready=ocr_enabled and ocr_available()
    if ocr_enabled and not ocr_ready:
        print("OCR: unavailable; continuing with native PDF text extraction.",flush=True)
    with pymupdf.open(pdf_path) as doc:
        for page_number,page in enumerate(doc,start=1):
            raw_text=clean_text(page.get_text("text"))
            image_count=len(page.get_images(full=True))
            text=raw_text
            extraction_method="pymupdf" if raw_text else "none"
            ocr_used=False
            ocr_error=None
            ocr_quality_score=None
            ocr_confidence=None
            ocr_text=""
            denoised=False
            denoising_model=None
            denoising_error=None
            denoising_corruption_ratio=0.0
            denoising_disagreement=0.0
            ocr_ran=False
            if ocr_ready and should_ocr(raw_text,image_count=image_count,min_chars=ocr_min_chars):
                try:
                    result=ocr_page_with_metadata(page,dpi=ocr_dpi or None)
                    ocr_text=clean_text(result["text"])
                    ocr_quality_score=result["quality_score"]
                    ocr_confidence=result.get("confidence")
                    ocr_ran=bool(ocr_text)
                    selected_method="ocr" if not raw_text else "pymupdf+ocr"
                    selected_text,_,selected_score=choose_text(raw_text,ocr_text)
                    if selected_text!=raw_text:
                        text=selected_text
                        extraction_method=selected_method
                        ocr_used=True
                except Exception as exc:
                    ocr_error=str(exc)
            if text:
                denoise=denoise_page(raw_text=text,ocr_text=ocr_text if ocr_ran else None,ocr_confidence=ocr_confidence,ocr_used=ocr_ran)
                if denoise["denoised"]:
                    text=clean_text(denoise["text"])
                    extraction_method=f"{extraction_method}+llm_denoise" if extraction_method!="none" else "llm_denoise"
                    denoised=True
                    ocr_used=ocr_used or ocr_ran
                denoising_model=denoise.get("model")
                denoising_error=denoise.get("error")
                denoising_corruption_ratio=denoise.get("corruption_ratio",0.0)
                denoising_disagreement=denoise.get("disagreement",0.0)
            text=remove_headers(text,header_candidates)
            metadata=build_page_metadata(page_number,raw_text,text,extraction_method,ocr_used,image_count,ocr_error)
            pages.append({"document_id":document_id,"source":source,"page_number":page_number,"text":text,"raw_text":raw_text,"ocr_text":ocr_text,"language":metadata["language"],"extraction_method":metadata["extraction_method"],"ocr_used":metadata["ocr_used"],"image_count":metadata["image_count"],"text_length":metadata["text_length"],"raw_text_length":metadata["raw_text_length"],"ocr_quality_score":ocr_quality_score,"ocr_confidence":ocr_confidence,"denoised":denoised,"denoising_model":denoising_model,"denoising_error":denoising_error,"denoising_corruption_ratio":denoising_corruption_ratio,"denoising_disagreement":denoising_disagreement,"ocr_error":metadata["ocr_error"]})
    return pages

if __name__=="__main__":
    pdf_path="data/documents/example.pdf"
    pages=parse_pdf(pdf_path,document_id="example_document")
    print(f"total pages: {len(pages)}")
    inspect_pages(pages,[1])