import io,os,re,shutil
from PIL import Image,ImageEnhance,ImageOps

OCR_LANGUAGES=os.getenv("OCR_LANGUAGES","eng+hin+mar")
OCR_DPI=int(os.getenv("OCR_DPI","300"))
OCR_PSM=os.getenv("OCR_PSM","6")
OCR_FALLBACK_PSM=os.getenv("OCR_FALLBACK_PSM","11")
OCR_MIN_TEXT_SCORE=float(os.getenv("OCR_MIN_TEXT_SCORE","0.42"))
OCR_IMAGE_TEXT_THRESHOLD=int(os.getenv("OCR_IMAGE_TEXT_THRESHOLD","900"))

def _tesseract():
    import pytesseract
    command=os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not command:
        raise RuntimeError("Tesseract OCR is not installed or TESSERACT_CMD is not configured.")
    pytesseract.pytesseract.tesseract_cmd=command
    return pytesseract

def ocr_available():
    try:
        _tesseract()
        return True
    except Exception:
        return False

def render_page(page,dpi=OCR_DPI):
    scale=float(dpi)/72.0
    import pymupdf
    pixmap=page.get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False)
    return Image.open(io.BytesIO(pixmap.tobytes("png")))

def preprocess_image(image):
    image=ImageOps.grayscale(image)
    image=ImageOps.autocontrast(image)
    image=ImageEnhance.Sharpness(image).enhance(1.35)
    return image

def detect_rotation(image):
    try:
        pytesseract=_tesseract()
        data=pytesseract.image_to_osd(image,config="--psm 0")
        match=re.search(r"Rotate:\s*(\d+)",data)
        return int(match.group(1)) if match else 0
    except Exception:
        return 0

def _clean_ocr(text):
    text=str(text or "").replace("\x0c","")
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

def text_quality_score(text):
    text=_clean_ocr(text)
    if not text:
        return 0.0
    chars=len(text)
    alpha=sum(c.isalpha() for c in text)
    alnum=sum(c.isalnum() for c in text)
    words=re.findall(r"[A-Za-z\u0900-\u097F]{2,}",text)
    good_words=sum(1 for w in words if len(w)>=2)
    bad_chars=sum(1 for c in text if c in "�￾")
    alpha_ratio=alpha/max(1,chars)
    word_density=min(1.0,len(words)/max(1,chars/6))
    good_ratio=good_words/max(1,len(words))
    noise=bad_chars/max(1,chars)
    return max(0.0,0.35*alpha_ratio+0.35*word_density+0.30*good_ratio-0.70*noise)

def should_ocr(text,image_count=0,min_chars=80):
    normalized=" ".join(str(text or "").split())
    if not normalized:
        return True
    if len(normalized)<min_chars:
        return True
    if image_count>0 and len(normalized)<OCR_IMAGE_TEXT_THRESHOLD:
        return True
    score=text_quality_score(normalized)
    if image_count>0 and score< OCR_MIN_TEXT_SCORE:
        return True
    return False

def ocr_page(page,lang=OCR_LANGUAGES,dpi=OCR_DPI):
    pytesseract=_tesseract()
    image=preprocess_image(render_page(page,dpi=dpi))
    rotation=detect_rotation(image)
    if rotation:
        image=image.rotate(rotation,expand=True)
    attempts=[]
    for psm in [OCR_PSM,OCR_FALLBACK_PSM]:
        try:
            text=_clean_ocr(pytesseract.image_to_string(image,lang=lang,config=f"--psm {psm}"))
            attempts.append((text_quality_score(text),text,psm))
        except Exception:
            continue
    if not attempts:
        raise RuntimeError("Tesseract OCR produced no usable output.")
    score,text,psm=max(attempts,key=lambda x:x[0])
    if not text:
        raise RuntimeError("Tesseract OCR returned empty text.")
    return text

def ocr_page_with_metadata(page,lang=OCR_LANGUAGES,dpi=OCR_DPI):
    pytesseract=_tesseract()
    image=preprocess_image(render_page(page,dpi=dpi))
    rotation=detect_rotation(image)
    if rotation:
        image=image.rotate(rotation,expand=True)
    attempts=[]
    for psm in [OCR_PSM,OCR_FALLBACK_PSM]:
        try:
            text=_clean_ocr(pytesseract.image_to_string(image,lang=lang,config=f"--psm {psm}"))
            data=pytesseract.image_to_data(image,lang=lang,config=f"--psm {psm}",output_type=pytesseract.Output.DICT)
            confidences=[]
            for value in data.get("conf",[]):
                try:
                    value=float(value)
                    if value>=0:
                        confidences.append(value)
                except (TypeError,ValueError):
                    continue
            confidence=sum(confidences)/len(confidences) if confidences else -1.0
            attempts.append((text_quality_score(text),text,psm,confidence))
        except Exception:
            continue
    if not attempts:
        raise RuntimeError("Tesseract OCR produced no usable output.")
    score,text,psm,confidence=max(attempts,key=lambda x:(x[0],x[3]))
    return {"text":text,"quality_score":score,"rotation":rotation,"psm":psm,"confidence":confidence}

def choose_text(raw_text,ocr_text):
    raw_score=text_quality_score(raw_text)
    ocr_score=text_quality_score(ocr_text)
    raw_len=len(str(raw_text or "").strip())
    ocr_len=len(str(ocr_text or "").strip())
    if not raw_text:
        return ocr_text,"ocr",ocr_score
    if not ocr_text:
        return raw_text,"pymupdf",raw_score
    if ocr_score>=raw_score+0.06:
        return ocr_text,"pymupdf+ocr",ocr_score
    if raw_len<80 and ocr_len>raw_len:
        return ocr_text,"pymupdf+ocr",ocr_score
    return raw_text,"pymupdf",raw_score
