from ingestion.language import detect_language

def build_page_metadata(page_number,raw_text,text,extraction_method,ocr_used,image_count=0,ocr_error=None):
    return {
        "page_number":page_number,
        "language":detect_language(text),
        "extraction_method":extraction_method,
        "ocr_used":bool(ocr_used),
        "image_count":int(image_count),
        "text_length":len(text),
        "raw_text_length":len(raw_text or ""),
        "ocr_error":ocr_error,
    }