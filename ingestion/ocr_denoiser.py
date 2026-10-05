import os,re
from typing import Any

OCR_DENOISER_ENABLED=os.getenv("OCR_DENOISER_ENABLED","true").lower() in {"1","true","yes","on"}
OCR_DENOISER_MODEL=os.getenv("OCR_DENOISER_MODEL","llama-3.1-8b-instant")
OCR_DENOISER_MIN_CHARS=int(os.getenv("OCR_DENOISER_MIN_CHARS","80"))
OCR_DENOISER_CONFIDENCE=float(os.getenv("OCR_DENOISER_CONFIDENCE","55"))
OCR_DENOISER_CORRUPTION=float(os.getenv("OCR_DENOISER_CORRUPTION","0.08"))
OCR_DENOISER_DISAGREEMENT=float(os.getenv("OCR_DENOISER_DISAGREEMENT","0.45"))
OCR_DENOISER_MAX_CHARS=int(os.getenv("OCR_DENOISER_MAX_CHARS","12000"))

SYSTEM_PROMPT="""You are SOURVETA's OCR text restoration engine.
The input is extracted text from a scanned, degraded, or handwritten document. Restore only OCR transcription errors.
STRICT RULES:
1. Make the smallest possible corrections to corrupted words, characters, spacing, and punctuation.
2. Preserve the original meaning, order, headings, bullets, numbering, line breaks, names, dates, numbers, units, identifiers, and symbols.
3. Never summarize, explain, answer questions, or add information.
4. Never infer missing facts from outside knowledge.
5. Never replace uncertain text with a domain-specific guess. If a token cannot be reconstructed confidently, preserve the original token.
6. Use alternative OCR text only as corroborating evidence from the same source page; never merge unrelated content.
7. Output ONLY the restored document text. No commentary."""

def _tokens(text:str)->list[str]:
    return re.findall(r"\S+",str(text or ""))

def corruption_ratio(text:str)->float:
    tokens=_tokens(text)
    if not tokens:
        return 0.0
    suspicious=0.0
    for token in tokens:
        if re.search(r"(?<=[A-Za-z])\d(?=[A-Za-z])|(?<=\d)[A-Za-z](?=\d)",token):
            suspicious+=1.0
        if re.search(r"[|_=~^]{2,}",token):
            suspicious+=1.0
        if re.search(r"[\x00-\x1f\x7f]",token):
            suspicious+=0.5
        if len(token)==1 and token.lower() not in {"a","i"} and not token.isdigit():
            suspicious+=0.5
    return min(1.0,suspicious/max(1,len(tokens)))

def token_disagreement(primary:str,alternate:str)->float:
    a={t.lower() for t in re.findall(r"[A-Za-z0-9\u0080-\uFFFF]+",primary or "") if len(t)>1}
    b={t.lower() for t in re.findall(r"[A-Za-z0-9\u0080-\uFFFF]+",alternate or "") if len(t)>1}
    if not a or not b:
        return 0.0
    return round(1.0-len(a & b)/max(1,len(a | b)),3)

def should_denoise(text:str,ocr_text:str|None=None,ocr_confidence:float|None=None,ocr_used:bool=False)->tuple[bool,float,float]:
    source=str(text or "").strip()
    if len(source)<OCR_DENOISER_MIN_CHARS:
        return False,0.0,0.0
    ratio=corruption_ratio(source)
    disagreement=token_disagreement(source,str(ocr_text or "")) if ocr_text else 0.0
    low_conf=ocr_confidence is not None and ocr_confidence>=0 and ocr_confidence< OCR_DENOISER_CONFIDENCE
    needs=ratio>=OCR_DENOISER_CORRUPTION or low_conf or (ocr_used and disagreement>=OCR_DENOISER_DISAGREEMENT and ocr_confidence is not None and ocr_confidence< OCR_DENOISER_CONFIDENCE+10)
    return needs,ratio,disagreement

def _digit_tokens(text:str)->list[str]:
    return re.findall(r"(?<![A-Za-z])\d+(?:[.,:/-]\d+)*(?![A-Za-z])",str(text or ""))

def _valid_candidate(source:str,candidate:str,alternate:str="")->bool:
    source=str(source or "").strip()
    candidate=str(candidate or "").strip()
    if not source or not candidate:
        return False
    if len(candidate)<max(20,int(len(source)*0.45)) or len(candidate)>int(len(source)*1.8):
        return False
    allowed_numbers=set(_digit_tokens(source)) | set(_digit_tokens(alternate))
    if any(token not in allowed_numbers for token in _digit_tokens(candidate)):
        return False
    source_lines=[x for x in source.splitlines() if x.strip()]
    candidate_lines=[x for x in candidate.splitlines() if x.strip()]
    if len(source_lines)>=4 and len(candidate_lines)<max(2,int(len(source_lines)*0.5)):
        return False
    if candidate.lower().startswith(("here is","here's","sure,","the corrected text")):
        return False
    return True

def _client():
    from groq import Groq
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)

def denoise_page(raw_text:str,ocr_text:str|None=None,ocr_confidence:float|None=None,ocr_used:bool=False)->dict[str,Any]:
    primary=str(raw_text or "").strip()
    alternate=str(ocr_text or "").strip()
    source=primary or alternate
    if not OCR_DENOISER_ENABLED or len(source)<OCR_DENOISER_MIN_CHARS:
        return {"text":source,"denoised":False,"corruption_ratio":corruption_ratio(source),"disagreement":0.0,"model":None,"error":None}
    needs,ratio,disagreement=should_denoise(source,alternate,ocr_confidence,ocr_used)
    if not needs:
        return {"text":source,"denoised":False,"corruption_ratio":ratio,"disagreement":disagreement,"model":None,"error":None}
    source=source[:OCR_DENOISER_MAX_CHARS]
    prompt=f"PRIMARY EXTRACTION FROM THE PAGE:\n{source}\n"
    if alternate and alternate!=source:
        prompt+=f"\nALTERNATIVE OCR EXTRACTION FROM THE SAME PAGE:\n{alternate[:OCR_DENOISER_MAX_CHARS]}\n"
    try:
        response=_client().chat.completions.create(
            model=OCR_DENOISER_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            temperature=0,
            max_completion_tokens=min(4096,max(512,len(source)//2)),
        )
        cleaned=(response.choices[0].message.content or "").strip()
        if not cleaned:
            raise RuntimeError("OCR denoiser returned empty text.")
        if not _valid_candidate(source,cleaned,alternate):
            raise RuntimeError("OCR denoiser candidate failed safety validation.")
        return {"text":cleaned,"denoised":True,"corruption_ratio":ratio,"disagreement":disagreement,"model":OCR_DENOISER_MODEL,"error":None}
    except Exception as exc:
        return {"text":source,"denoised":False,"corruption_ratio":ratio,"disagreement":disagreement,"model":OCR_DENOISER_MODEL,"error":str(exc)}
