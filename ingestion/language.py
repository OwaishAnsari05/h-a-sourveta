import re,unicodedata

MARATHI_HINTS={"आहे","आणि","मध्ये","म्हणून","करण्यासाठी","यांच्या","तसेच","होते","असलेले","महाराष्ट्र","आपण","त्यामुळे","यामुळे","करण्यात","झालेले","असून","यांना"}
HINDI_HINTS={"है","और","में","के","की","का","था","थी","थे","होता","होती","यह","से","लिए","तथा","लेकिन","द्वारा","करने","किया","गया","गई"}

def normalize_unicode(text):
    text=unicodedata.normalize("NFKC",str(text or ""))
    text=text.replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\ufeff","")
    return text

def _script_counts(text):
    devanagari=sum(1 for c in text if "DEVANAGARI" in unicodedata.name(c,""))
    latin=sum(1 for c in text if "LATIN" in unicodedata.name(c,""))
    return devanagari,latin

def detect_language(text):
    text=normalize_unicode(text)
    if not text.strip():
        return "unknown"
    devanagari,latin=_script_counts(text)
    total=max(1,sum(1 for c in text if c.isalpha()))
    dev_ratio=devanagari/total
    lat_ratio=latin/total
    if dev_ratio<0.15 and lat_ratio>=0.50:
        return "eng"
    if dev_ratio>=0.50:
        tokens=set(re.findall(r"[\u0900-\u097F]+",text))
        marathi_score=sum(1 for token in tokens if token in MARATHI_HINTS)
        hindi_score=sum(1 for token in tokens if token in HINDI_HINTS)
        if marathi_score>hindi_score and marathi_score>0:
            return "mar"
        if hindi_score>marathi_score and hindi_score>0:
            return "hin"
        if any(c in text for c in "ळऴॲऑ"):
            return "mar"
        return "hin"
    if devanagari>0 and latin>0:
        return "mixed"
    return "unknown"
