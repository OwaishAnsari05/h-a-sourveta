import json,os,re
from functools import lru_cache
from typing import Any,TypedDict
from difflib import SequenceMatcher
from rank_bm25 import BM25Okapi
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from generation.citation import build_citations,validate_citations,format_citations as format_structured_citations,extract_answer_numbers,extract_numbers,normalize_number
from vectorstore.reranker import rerank_documents

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH=os.path.join(PROJECT_ROOT,"data","chunks.json")
DOCUMENT_CHUNKS_DIR=os.path.join(PROJECT_ROOT,"data","chunks")
CHROMA_PATH=os.path.join(PROJECT_ROOT,"data","chroma_db")
COLLECTION_NAME="sourveta_multilingual"
EMBEDDING_MODEL="intfloat/multilingual-e5-base"
GROQ_MODEL="openai/gpt-oss-20b"
VECTOR_TOP_K=15
BM25_TOP_K=15
HYBRID_TOP_K=15
RERANK_TOP_K=12
CONTEXT_TOP_K=5
RRF_K=60
VECTOR_WEIGHT=1.0
BM25_WEIGHT=1.0
LIGHTWEIGHT_RAG=os.getenv("LIGHTWEIGHT_RAG","0")=="1"


STOPWORDS={
    "what","was","were","the","is","are","as","of","in","on","for",
    "to","and","a","an","at","by","from","with","who","which","how",
    "much","many","did","do","does","this","that","these","those",
    "year","date","march","31","2025","2024","fy","between","happened",
    "company","companies","reported","report","say","said","does",
}

REFUSAL="I could not find sufficiently relevant information in the document to answer this question."

class RAGResponse(TypedDict):
    answer:str
    sources:list[dict[str,Any]]
    query:str
    document_id:str|None
    source_count:int

def load_chunks():
    if not os.path.exists(CHUNKS_PATH): return []
    with open(CHUNKS_PATH,"r",encoding="utf-8") as file: return json.load(file)

CHUNKS=load_chunks()

@lru_cache(maxsize=32)
def load_document_chunks(document_id:str):
    path=os.path.join(DOCUMENT_CHUNKS_DIR,f"{document_id}.json")
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as file:
            return json.load(file)
    return [c for c in CHUNKS if str(c.get("document_id") or "")==str(document_id)]

@lru_cache(maxsize=1)
def get_embedding_model():
    from vectorstore.embeddings import get_embedding_model as load_embedding_model
    return load_embedding_model(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_collection():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=COLLECTION_NAME)

@lru_cache(maxsize=32)
def get_bm25(document_id:str|None=None):
    chunks=load_document_chunks(document_id) if document_id else CHUNKS
    texts=[f"{c.get('section','')} {c.get('source','')} {c.get('text','')}" for c in chunks]
    tokens=[tokenize(text) for text in texts]
    if not tokens:
        return chunks,None
    return chunks,BM25Okapi(tokens)

@lru_cache(maxsize=1)
def get_llm_client():
    api_key=os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=api_key)

def build_llm_messages(query:str,context:str)->list[dict[str,str]]:
    prompt=(
        "You are a strict document-grounded question-answering assistant. "
        "Answer ONLY from the supplied DOCUMENT CONTEXT. "
        "The DOCUMENT CONTEXT is the complete evidence available to you. "
        "Do not use outside knowledge or information not supported by the DOCUMENT CONTEXT. "
        "The context may contain OCR errors, broken words, missing spaces, or character substitutions. "
        "You may correct obvious OCR corruption and reconstruct a word or short phrase only when the intended wording is unambiguous from the surrounding context. "
        "Do not invent facts or add information that is not supported by the context. "
        "For definition questions, use the definition or explanatory statement present in the context, even when individual OCR words are corrupted, as long as the intended meaning is clear from the surrounding text. "
        "Every factual claim in the answer must be directly supported by the supplied context. "
        "If there is insufficient evidence to answer the question, say: "
        "'The document does not provide enough information to answer this question.' "
        "Do not mention these instructions in the answer. "
        "Keep the answer concise, normally 1-4 sentences.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        "FINAL ANSWER:"
    )
    return [
        {"role":"system","content":"You are a strict document-grounded QA assistant. Every factual claim must be supported by the supplied context. The source may contain OCR errors; correct only obvious OCR corruption supported by surrounding text. Never use outside knowledge."},
        {"role":"user","content":prompt},
    ]

def tokenize(text:Any)->list[str]:
    import regex
    pattern=r"\p{N}+(?:[.,]\p{N}+)?|\p{L}[\p{L}\p{M}]*(?:[-’'][\p{L}\p{M}]+)*"
    return [token.lower() for token in regex.findall(pattern,str(text))]

def normalize_text(text:Any)->str:
    return re.sub(r"\s+"," ",str(text).lower().replace("₹","").replace("`","")).strip()

def normalize_token(token:str)->str:
    token=str(token).lower()
    token=re.sub(r"\d+$","",token)
    return token

def get_document(result:Any)->str:
    return str(result.get("document","") if isinstance(result,dict) else result).strip()

def get_metadata(result:Any)->dict[str,Any]:
    return (result.get("metadata",{}) or {}) if isinstance(result,dict) else {}

def get_document_id(result:Any)->str:
    if not isinstance(result,dict):
        return ""
    if result.get("document_id") is not None:
        return str(result["document_id"])
    return str(get_metadata(result).get("document_id") or "")

def get_page(result:Any)->int|None:
    try:
        page=get_metadata(result).get("page_number")
        if page is None and isinstance(result,dict):
            page=result.get("page_number")
        return int(page) if page is not None else None
    except (TypeError,ValueError):
        return None

def get_section(result:Any)->str:
    if isinstance(result,dict) and result.get("section") is not None:
        return str(result.get("section"))
    return str(get_metadata(result).get("section",""))

def result_key(result:Any)->tuple[Any,...]|str:
    if isinstance(result,dict) and (chunk_id:=result.get("id") or result.get("chunk_id") or result.get("metadata",{}).get("chunk_id")):
        return (get_document_id(result),str(chunk_id))
    return (get_document_id(result),get_page(result),get_section(result),get_document(result))

def deduplicate_results(results:list[dict[str,Any]]|None)->list[dict[str,Any]]:
    unique,seen=[],set()
    for result in results or []:
        key=result_key(result)
        if key not in seen:
            seen.add(key)
            unique.append(result)
    return unique

def minmax_normalize(values:list[float])->list[float]:
    if not values:
        return []
    low,high=min(values),max(values)
    if high<=low:
        return [1.0 if high>0 else 0.0 for _ in values]
    return [(value-low)/(high-low) for value in values]

def contains_number(text:str,value:str)->bool:
    normalized=normalize_text(text).replace(",","")
    return value.replace(",","").lower() in normalized

def has_phrase(text:str,phrase:str)->bool:
    return normalize_text(phrase) in normalize_text(text)

def merge_result_records(results:list[dict[str,Any]])->list[dict[str,Any]]:
    merged={}
    for result in results:
        key=result_key(result)
        if key not in merged:
            merged[key]=result.copy()
            continue
        current=merged[key]
        for field in ("rrf_score","vector_rrf","bm25_rrf","evidence_rescue_score","bm25_score","distance"):
            if field in result:
                current[field]=max(float(current.get(field,0.0)),float(result.get(field,0.0)))
        for field in ("vector_rank","bm25_rank"):
            if current.get(field) is None and result.get(field) is not None:
                current[field]=result[field]
        if result.get("evidence_matches"):
            current["evidence_matches"]=list(dict.fromkeys(list(current.get("evidence_matches",[]))+list(result["evidence_matches"])))
        if not current.get("document") and result.get("document"):
            current["document"]=result["document"]
        if not current.get("metadata") and result.get("metadata"):
            current["metadata"]=result["metadata"]
        if not current.get("document_id") and result.get("document_id"):
            current["document_id"]=result["document_id"]
    return list(merged.values())

def query_profile(query:str)->dict[str,Any]:
    normalized=normalize_text(query)
    return {"type":"general","comparative":any(t in normalized for t in ("between","change","changed","increase","decrease","growth","difference","compared","comparison"))}

def retrieve_vector(query:str,collection:Any,embed_model:Any,top_k:int=VECTOR_TOP_K,document_id:str|None=None)->list[dict[str,Any]]:
    from vectorstore.embeddings import encode_query
    query_kwargs={
        "query_embeddings":[encode_query(embed_model,query).tolist()],
        "n_results":top_k,
        "include":["documents","metadatas","distances"],
    }
    if document_id:
        query_kwargs["where"]={"document_id":document_id}
    response=collection.query(**query_kwargs)
    docs=response.get("documents",[[]])[0]
    metas=response.get("metadatas",[[]])[0]
    dists=response.get("distances",[[]])[0]
    ids=response.get("ids",[[]])[0]
    return [
        {
            "id":ids[i] if i<len(ids) else (metadata or {}).get("chunk_id",""),
            "document":document,
            "metadata":metadata or {},
            "document_id":(metadata or {}).get("document_id",""),
            "distance":float(dists[i]) if i<len(dists) else 0.0,
        }
        for i,(document,metadata) in enumerate(zip(docs,metas))
    ]

def retrieve_bm25(query:str,chunks:list[dict[str,Any]],bm25_index:Any,top_k:int=BM25_TOP_K)->list[dict[str,Any]]:
    if not (tokens:=tokenize(query)) or bm25_index is None:
        return []
    scores=bm25_index.get_scores(tokens)
    return [
        {
            "id":chunks[int(i)].get("chunk_id",""),
            "document":chunks[int(i)].get("text",""),
            "document_id":chunks[int(i)].get("document_id",""),
            "metadata":{
                key:chunks[int(i)].get(key,"Unknown" if key in {"source","section"} else (0 if "index" in key else None))
                for key in ("chunk_id","document_id","source","page_number","section","section_index","chunk_index","language","extraction_method","ocr_used")
            },
            "bm25_score":float(scores[int(i)]),
        }
        for i in scores.argsort()[::-1][:top_k]
    ]

def reciprocal_rank_fusion(vector_results:list[dict[str,Any]],bm25_results:list[dict[str,Any]],top_k:int=HYBRID_TOP_K)->list[dict[str,Any]]:
    fused={}
    for prefix,results,weight in (("vector",vector_results,VECTOR_WEIGHT),("bm25",bm25_results,BM25_WEIGHT)):
        for rank,result in enumerate(results,start=1):
            key=result_key(result)
            if key not in fused:
                fused[key]={
                    "id":result.get("id") or get_metadata(result).get("chunk_id",""),
                    "document":get_document(result),
                    "document_id":get_document_id(result),
                    "metadata":get_metadata(result),
                    "vector_rrf":0.0,
                    "bm25_rrf":0.0,
                    "rrf_score":0.0,
                    "vector_rank":None,
                    "bm25_rank":None,
                }
            rrf=weight/(RRF_K+rank)
            fused[key][f"{prefix}_rrf"]=rrf
            fused[key][f"{prefix}_rank"]=rank
            fused[key]["rrf_score"]+=rrf
    return sorted(fused.values(),key=lambda item:item["rrf_score"],reverse=True)[:top_k]

def retrieve_evidence_rescue(query:str,chunks:list[dict[str,Any]],max_results:int=10)->list[dict[str,Any]]:
    rescued=[]
    generic_terms=generic_query_terms(query)
    for chunk in chunks:
        doc=normalize_text(chunk.get("text","")); sec=normalize_text(chunk.get("section",""))
        document_tokens=[normalize_token(token) for token in tokenize(doc)]
        section_tokens=[normalize_token(token) for token in tokenize(sec)]
        doc_overlap=fuzzy_overlap_score(generic_terms,document_tokens,threshold=0.70)
        section_overlap=fuzzy_overlap_score(generic_terms,section_tokens,threshold=0.70)
        phrase_overlap=phrase_overlap_score(generic_terms,doc)
        concept_score=max(doc_overlap,section_overlap,phrase_overlap)
        required=0.55 if len(generic_terms)>=2 else 0.70
        if concept_score<required: continue
        score=concept_score*100.0+section_overlap*20.0+phrase_overlap*15.0
        matches=[f"term_overlap:{doc_overlap:.2f}"]
        if section_overlap>0: matches.append(f"section_overlap:{section_overlap:.2f}")
        if phrase_overlap>0: matches.append(f"phrase_overlap:{phrase_overlap:.2f}")
        rescued.append({"id":chunk.get("chunk_id",""),"document":chunk.get("text",""),"document_id":chunk.get("document_id",""),"metadata":{key:chunk.get(key,"Unknown" if key in {"source","section"} else 0) for key in ("chunk_id","document_id","source","page_number","section","section_index","chunk_index","language","extraction_method","ocr_used")},"evidence_rescue_score":score,"evidence_matches":matches,"rrf_score":0.0,"vector_rank":None,"bm25_rank":None})
    return sorted(rescued,key=lambda item:item["evidence_rescue_score"],reverse=True)[:max_results]

def hybrid_retrieve(query:str,top_k:int=HYBRID_TOP_K,document_id:str|None=None)->list[dict[str,Any]]:
    print(" HYBRID: START ",flush=True)
    chunks,bm25_index=get_bm25(document_id)
    print(f" HYBRID: BM25 LOADED ({len(chunks)} CHUNKS) ",flush=True)

    if LIGHTWEIGHT_RAG:
        print(" HYBRID: LIGHTWEIGHT MODE ",flush=True)
        print(" HYBRID: BM25 RETRIEVAL START ",flush=True)
        bm25_results=retrieve_bm25(query,chunks,bm25_index,BM25_TOP_K)
        print(f" HYBRID: BM25 RETRIEVAL END ({len(bm25_results)} RESULTS) ",flush=True)

        print(" HYBRID: EVIDENCE RESCUE START ",flush=True)
        rescued=retrieve_evidence_rescue(query,chunks,10)
        print(f" HYBRID: EVIDENCE RESCUE END ({len(rescued)} RESULTS) ",flush=True)

        merged=merge_result_records(bm25_results+rescued)

        if document_id:
            merged=[r for r in merged if get_document_id(r)==str(document_id)]

        result=sorted(
            merged,
            key=lambda item:(
                float(item.get("evidence_rescue_score",0.0))>0,
                float(item.get("evidence_rescue_score",0.0)),
                float(item.get("bm25_score",0.0)),
            ),
            reverse=True,
        )[:top_k]

        print(f" HYBRID: LIGHTWEIGHT FINISHED ({len(result)} RESULTS) ",flush=True)
        return result

    print(" HYBRID: LOADING EMBEDDING MODEL ",flush=True)
    embed_model=get_embedding_model()
    print(" HYBRID: EMBEDDING MODEL LOADED ",flush=True)

    print(" HYBRID: LOADING CHROMA COLLECTION ",flush=True)
    collection=get_collection()
    print(" HYBRID: CHROMA COLLECTION LOADED ",flush=True)

    print(" HYBRID: VECTOR RETRIEVAL START ",flush=True)
    vector_results=retrieve_vector(query,collection,embed_model,VECTOR_TOP_K,document_id)
    print(f" HYBRID: VECTOR RETRIEVAL END ({len(vector_results)} RESULTS) ",flush=True)

    print(" HYBRID: BM25 RETRIEVAL START ",flush=True)
    bm25_results=retrieve_bm25(query,chunks,bm25_index,BM25_TOP_K)
    print(f" HYBRID: BM25 RETRIEVAL END ({len(bm25_results)} RESULTS) ",flush=True)

    print(" HYBRID: RRF START ",flush=True)
    fused=reciprocal_rank_fusion(vector_results,bm25_results,top_k)
    print(f" HYBRID: RRF END ({len(fused)} RESULTS) ",flush=True)

    print(" HYBRID: EVIDENCE RESCUE START ",flush=True)
    rescued=retrieve_evidence_rescue(query,chunks,10)
    print(f" HYBRID: EVIDENCE RESCUE END ({len(rescued)} RESULTS) ",flush=True)

    merged=merge_result_records(fused+rescued)

    if document_id:
        merged=[r for r in merged if get_document_id(r)==str(document_id)]

    result=sorted(
        merged,
        key=lambda item:(
            float(item.get("evidence_rescue_score",0.0))>0,
            float(item.get("evidence_rescue_score",0.0)),
            float(item.get("rrf_score",0.0)),
        ),
        reverse=True,
    )[:top_k]

    print(f" HYBRID: FINISHED ({len(result)} RESULTS) ",flush=True)
    return result

def add_lexical_scores(results:list[dict[str,Any]],query:str)->list[dict[str,Any]]:
    query_terms=[normalize_token(token) for token in tokenize(query) if token.isalpha() and token not in STOPWORDS and len(token)>=3]
    scored=[]
    for raw in results or []:
        item=raw.copy(); doc=normalize_text(get_document(item)); sec=normalize_text(get_section(item))
        sec_tokens=set(tokenize(sec)); doc_tokens=set(tokenize(doc))
        score=sum(3.0 if term in sec_tokens else 1.0 if term in doc_tokens else 0.0 for term in query_terms)
        item["lexical_score"]=score; scored.append(item)
    return scored

def calculate_evidence_features(query:str,item:dict[str,Any])->dict[str,Any]:
    sec=normalize_text(get_section(item)); doc=normalize_text(get_document(item))
    query_terms=generic_query_terms(query)
    document_tokens=[normalize_token(token) for token in tokenize(doc)]
    section_tokens=[normalize_token(token) for token in tokenize(sec)]
    query_overlap=fuzzy_overlap_score(query_terms,document_tokens,threshold=0.65)
    section_overlap=fuzzy_overlap_score(query_terms,section_tokens,threshold=0.65)
    phrase_overlap=phrase_overlap_score(query_terms,doc)
    evidence_score=max(query_overlap,phrase_overlap,section_overlap)
    return {"page_evidence_score":0.0,"evidence_match_score":min(evidence_score,1.0),"exact_evidence":evidence_score>=0.65,"mismatch_penalty":0.0,"section_match":section_overlap>=0.65,"phrase_match":phrase_overlap>0.0,"value_match":False}

def page_aware_rerank(query:str,results:list[dict[str,Any]],top_k:int=RERANK_TOP_K)->list[dict[str,Any]]:
    if not results:
        return []
    results=add_lexical_scores(deduplicate_results(results),query)
    documents=[get_document(result) for result in results]
    reranked_pairs=rerank_documents(query,documents,top_k=len(documents))
    by_document={}
    for result in results:
        by_document.setdefault(get_document(result),[]).append(result)
    reranked=[]
    for ranked in reranked_pairs:
        document=ranked.get("document","") if isinstance(ranked,dict) else ranked[0]
        score=ranked.get("rerank_score",ranked.get("score",0.0)) if isinstance(ranked,dict) else ranked[1]
        if by_document.get(document):
            candidate=by_document[document].pop(0).copy()
            candidate["rerank_score"]=float(score)
            reranked.append(candidate)
    rerank_values=[float(r.get("rerank_score",0.0)) for r in reranked]
    lexical_values=[float(r.get("lexical_score",0.0)) for r in reranked]
    rerank_norms=minmax_normalize(rerank_values)
    lexical_norms=minmax_normalize(lexical_values)
    for index,item in enumerate(reranked):
        features=calculate_evidence_features(query,item)
        item["rerank_norm"]=rerank_norms[index]
        item["lexical_norm"]=lexical_norms[index]
        item.update(features)
        item["entity_score"]=0.0
        item["relationship_score"]=0.0
        rescue_score=float(item.get("evidence_rescue_score",0.0))
        rescue_norm=min(rescue_score/200.0,1.0)
        item["final_score"]=(
            rerank_norms[index]*0.35+
            lexical_norms[index]*0.20+
            features["page_evidence_score"]*0.15+
            features["evidence_match_score"]*0.30+
            rescue_norm*0.10-
            features["mismatch_penalty"]
        )
        if features["exact_evidence"]:
            item["final_score"]+=0.50
    return sorted(
        reranked,
        key=lambda item:(
            bool(item.get("exact_evidence",False)),
            item.get("final_score",0.0),
            item.get("evidence_match_score",0.0),
            item.get("rerank_score",0.0),
        ),
        reverse=True,
    )[:top_k]

def rerank_candidates(query:str,candidates:list[dict[str,Any]],top_k:int=RERANK_TOP_K)->list[dict[str,Any]]:
    return page_aware_rerank(query,candidates,top_k)

def protect_exact_matches(
    query:str,
    reranked_results:list[dict[str,Any]],
    candidates:list[dict[str,Any]]|None=None,
    candidate_results:list[dict[str,Any]]|None=None,
    top_k:int=RERANK_TOP_K,
)->list[dict[str,Any]]:
    source_candidates=candidates if candidates is not None else candidate_results
    merged=merge_result_records((reranked_results or [])+(source_candidates or []))
    for item in merged:
        features=calculate_evidence_features(query,item)
        item.update(features)
        if features["exact_evidence"]:
            item["final_score"]=max(float(item.get("final_score",0.0)),0.90)
    return sorted(
        merged,
        key=lambda item:(
            bool(item.get("exact_evidence",False)),
            item.get("final_score",0.0),
            item.get("evidence_match_score",0.0),
            item.get("rerank_score",0.0),
        ),
        reverse=True,
    )[:top_k]

def filter_context_results(query:str,results:list[dict[str,Any]],min_score:float=0.35)->list[dict[str,Any]]:
    return filter_generic_context_results(query,results) if results else []

def generic_query_terms(query:str)->list[str]:
    base=[normalize_token(token) for token in tokenize(query) if token.isalpha() and token not in STOPWORDS and len(token)>=3]
    expansions={"types":["kind","category","classification"],"type":["kind","category","classification"],"difference":["different","comparison"],"differences":["different","comparison"]}
    terms=list(dict.fromkeys(base))
    for term in base:
        terms.extend(expansions.get(term,[]))
    return list(dict.fromkeys(terms))
def generic_answer_terms(answer:str)->list[str]:
    units={"lakh","lakhs","crore","crores","million","billion"}
    return [normalize_token(token) for token in tokenize(answer) if token.isalpha() and token not in STOPWORDS and token not in units and len(token)>=3]
def fuzzy_token_match(term:str,candidate:str)->float:
    term=normalize_token(term)
    candidate=normalize_token(candidate)
    if not term or not candidate:
        return 0.0
    if term==candidate:
        return 1.0
    if term in candidate or candidate in term:
        return min(len(term),len(candidate))/max(len(term),len(candidate))
    if len(term)>=5 and len(candidate)>=5:
        return SequenceMatcher(None,term,candidate).ratio()
    return 0.0

def fuzzy_overlap_score(terms:list[str],document_tokens:list[str],threshold:float=0.82)->float:
    if not terms:
        return 0.0
    hits=0
    for term in set(terms):
        best=max((fuzzy_token_match(term,candidate) for candidate in document_tokens),default=0.0)
        if best>=threshold:
            hits+=1
    return hits/max(len(set(terms)),1)

def ordered_term_match(terms:list[str],tokens:list[str])->bool:
    position=-1
    for term in terms:
        matches=[i for i,token in enumerate(tokens) if i>position and fuzzy_token_match(term,token)>=0.82]
        if not matches:
            return False
        position=matches[0]
    return True

def phrase_overlap_score(terms:list[str],text:str)->float:
    if not terms:
        return 0.0
    normalized=normalize_text(text)
    hits=0
    for term in set(terms):
        if len(term)>=5 and term in normalized:
            hits+=1
    return hits/max(len(set(terms)),1)

def generic_citation_score(query:str,answer:str,result:dict[str,Any])->float:
    query_terms=generic_query_terms(query)
    answer_terms=generic_answer_terms(answer)
    document_tokens=[normalize_token(token) for token in tokenize(get_document(result))]
    section_tokens=[normalize_token(token) for token in tokenize(get_section(result))]
    document_text=normalize_text(get_document(result))
    section_text=normalize_text(get_section(result))
    if not query_terms and not answer_terms:
        return 0.0
    query_overlap=fuzzy_overlap_score(query_terms,document_tokens)
    answer_overlap=fuzzy_overlap_score(answer_terms,document_tokens)
    section_query_overlap=fuzzy_overlap_score(query_terms,section_tokens)
    section_answer_overlap=fuzzy_overlap_score(answer_terms,section_tokens)
    query_phrase_overlap=phrase_overlap_score(query_terms,document_text)
    answer_phrase_overlap=phrase_overlap_score(answer_terms,document_text)
    existing_relevance=float(result.get("generic_relevance_score",0.0))
    lexical_norm=float(result.get("lexical_norm",0.0))
    rerank_norm=float(result.get("rerank_norm",0.0))
    rrf_norm=min(float(result.get("rrf_score",0.0))*100.0,1.0)
    concept_support=max(query_overlap,query_phrase_overlap,section_query_overlap)
    answer_support=max(answer_overlap,answer_phrase_overlap,section_answer_overlap)
    if query_terms and concept_support<0.45:
        return 0.0
    return (
        concept_support*0.42+
        answer_support*0.28+
        query_phrase_overlap*0.10+
        answer_phrase_overlap*0.08+
        section_query_overlap*0.04+
        section_answer_overlap*0.02+
        existing_relevance*0.03+
        lexical_norm*0.015+
        rerank_norm*0.01+
        rrf_norm*0.005
    )

def build_generic_evidence_preview(query:str,answer:str,result:dict[str,Any],max_chars:int=320)->str:
    document=re.sub(r"\s+"," ",get_document(result)).strip()
    if not document:
        return ""
    query_terms=generic_query_terms(query)
    answer_terms=generic_answer_terms(answer)
    sentences=re.split(r"(?<=[.!?])\s+",document)
    if len(sentences)==1 and len(document)>max_chars:
        words=document.split()
        windows=[]
        window_size=45
        for index in range(0,len(words),max(1,window_size//2)):
            window=" ".join(words[index:index+window_size])
            if window:
                windows.append(window)
        sentences=windows
    best_sentence=""
    best_score=-1.0
    for sentence in sentences:
        sentence_tokens=[normalize_token(token) for token in tokenize(sentence)]
        query_score=fuzzy_overlap_score(query_terms,sentence_tokens,threshold=0.65)
        answer_score=fuzzy_overlap_score(answer_terms,sentence_tokens,threshold=0.65)
        phrase_score=phrase_overlap_score(query_terms,sentence)
        concept_score=max(query_score,phrase_score)
        if query_terms and concept_score<0.35:
            continue
        score=concept_score*0.60+answer_score*0.30+min(phrase_score,1.0)*0.10
        if score>best_score:
            best_score=score
            best_sentence=sentence
    if not best_sentence:
        return ""
    evidence=best_sentence
    if len(evidence)>max_chars:
        evidence=evidence[:max_chars].rsplit(" ",1)[0]+"..."
    return evidence

def format_generic_citations(query:str,answer:str,results:list[dict[str,Any]],max_citations:int=3)->str:
    if not results:
        return ""
    candidates=[]
    seen=set()
    for result in deduplicate_results(results):
        metadata=get_metadata(result)
        document_id=get_document_id(result)
        source=str(metadata.get("source") or "Unknown")
        page=metadata.get("page_number","Unknown")
        section=str(metadata.get("section") or "").strip()
        key=(document_id,page,section)
        if key in seen:
            continue
        seen.add(key)
        score=generic_citation_score(query,answer,result)
        evidence=build_generic_evidence_preview(query,answer,result)
        if score<=0.0 or not evidence:
            continue
        candidates.append((score,result,evidence))
    candidates.sort(
        key=lambda item:(
            item[0],
            float(item[1].get("generic_relevance_score",0.0)),
            float(item[1].get("lexical_norm",0.0)),
            float(item[1].get("rerank_norm",0.0)),
            float(item[1].get("rrf_score",0.0)),
        ),
        reverse=True,
    )
    citations=[]
    for _,result,evidence in candidates[:max_citations]:
        metadata=get_metadata(result)
        source=str(metadata.get("source") or "Unknown")
        page=metadata.get("page_number","Unknown")
        section=str(metadata.get("section") or "").strip()
        citation=f"[C{len(citations)+1}] {source} | Page {page}"
        if section and section!="Unknown":
            citation+=f" | Section: {section}"
        citation+=f"\n    Evidence: {evidence}"
        citations.append(citation)
    return "\n".join(citations)

def filter_generic_context_results(query:str,results:list[dict[str,Any]],max_results:int=CONTEXT_TOP_K)->list[dict[str,Any]]:
    if not results:
        return []
    terms=generic_query_terms(query)
    scored=[]
    for result in results:
        document=normalize_text(get_document(result))
        section=normalize_text(get_section(result))
        lexical_norm=float(result.get("lexical_norm",0.0))
        rerank_norm=float(result.get("rerank_norm",0.0))
        rerank_score=float(result.get("rerank_score",0.0))
        rrf_score=float(result.get("rrf_score",0.0))
        rescue_score=float(result.get("evidence_rescue_score",0.0))
        document_tokens=[normalize_token(token) for token in tokenize(document)]
        section_tokens=[normalize_token(token) for token in tokenize(section)]
        term_hits=fuzzy_overlap_score(terms,document_tokens,threshold=0.65)
        section_hits=fuzzy_overlap_score(terms,section_tokens,threshold=0.65)
        phrase_hits=phrase_overlap_score(terms,document)
        evidence_match=max(term_hits,section_hits,phrase_hits)
        relevance=(
            evidence_match*0.55+
            section_hits*0.10+
            phrase_hits*0.10+
            lexical_norm*0.08+
            rerank_norm*0.10+
            min(rrf_score*100.0,1.0)*0.02+
            min(rescue_score/100.0,1.0)*0.05
        )
        if evidence_match>=0.30 or rescue_score>0 or bool(result.get("exact_evidence",False)):
            item=result.copy()
            item["generic_relevance_score"]=relevance
            item["generic_evidence_score"]=evidence_match
            scored.append(item)
    if not scored:
        return []
    scored.sort(
        key=lambda item:(
            float(item.get("generic_relevance_score",0.0)),
            float(item.get("generic_evidence_score",0.0)),
            bool(item.get("exact_evidence",False)),
            float(item.get("rerank_norm",0.0)),
            float(item.get("lexical_norm",0.0)),
            float(item.get("rrf_score",0.0)),
        ),
        reverse=True,
    )
    strongest=float(scored[0].get("generic_relevance_score",0.0))
    filtered=[item for item in scored if float(item.get("generic_relevance_score",0.0))>=max(0.18,strongest*0.35)]
    return filtered[:max_results]

def build_context(
    results:list[dict[str,Any]],
    hybrid_results:list[dict[str,Any]]|None=None,
    query:str|None=None,
    top_k:int=CONTEXT_TOP_K,
)->tuple[str,list[dict[str,Any]]]:
    if not results and not hybrid_results:
        return "",[]
    merged=merge_result_records((results or [])+(hybrid_results or []))
    if query:
        merged=protect_exact_matches(query,merged,top_k=max(top_k,len(merged)))
        merged=filter_context_results(query,merged)
    selected=sorted(
        merged,
        key=lambda item:(
            bool(item.get("exact_evidence",False)),
            item.get("final_score",0.0),
            item.get("evidence_match_score",0.0),
            item.get("rerank_score",0.0),
            item.get("generic_relevance_score",0.0),
        ),
        reverse=True,
    )[:top_k]
    ctx_parts=[]
    for index,result in enumerate(selected,start=1):
        metadata=get_metadata(result)
        scores=[]
        if result.get("rerank_score") is not None:
            scores.append(f"Rerank Score: {float(result['rerank_score']):.4f}")
        if result.get("rerank_norm") is not None:
            scores.append(f"RerankNorm: {float(result['rerank_norm']):.3f}")
        if result.get("lexical_score") is not None:
            scores.append(f"Lexical: {float(result['lexical_score']):.2f}")
        if result.get("lexical_norm") is not None:
            scores.append(f"LexNorm: {float(result['lexical_norm']):.3f}")
        if result.get("generic_relevance_score") is not None:
            scores.append(f"GenericRel: {float(result['generic_relevance_score']):.3f}")
        if result.get("final_score") is not None:
            scores.append(f"Final: {float(result['final_score']):.4f}")
        scores.append(f"PageEv: {float(result.get('page_evidence_score',0.0)):.3f}")
        scores.append(f"Evidence: {float(result.get('evidence_match_score',0.0)):.3f}")
        scores.append(f"Exact: {result.get('exact_evidence',False)}")
        ctx_parts.append(
            f"SOURCE {index}\n"
            f"Document ID: {get_document_id(result)}\n"
            f"Page: {metadata.get('page_number','Unknown')}\n"
            f"Source: {metadata.get('source','Unknown')}\n"
            f"Section: {metadata.get('section','Unknown')}\n"
            f"{' | '.join(scores)}\n"
            f"Content:\n{get_document(result)}\n"
        )
    return "\n".join(ctx_parts),selected

def generate_llm_answer(query:str,context:str)->str:
    client=get_llm_client()
    print("========== GENERATION CONTEXT ==========",flush=True)
    print(context[:12000],flush=True)
    print("========== END GENERATION CONTEXT ==========",flush=True)
    response=client.chat.completions.create(
        model=GROQ_MODEL,
        messages=build_llm_messages(query,context),
        temperature=0,
        max_completion_tokens=1024,
        stream=False,
        include_reasoning=False,
        reasoning_effort="low",
    )
    print(f"GROQ MODEL: {GROQ_MODEL}",flush=True)
    print(f"GROQ FINISH: {response.choices[0].finish_reason}",flush=True)
    print(f"GROQ CONTENT: {response.choices[0].message.content!r}",flush=True)
    print(f"GROQ REASONING: {getattr(response.choices[0].message,'reasoning',None)!r}",flush=True)
    answer=response.choices[0].message.content or ""
    return re.sub(r"^FINAL ANSWER:\s*","",answer.strip(),flags=re.IGNORECASE).strip()

def stream_llm_answer(query:str,context:str):
    client=get_llm_client()
    stream=client.chat.completions.create(
        model=GROQ_MODEL,
        messages=build_llm_messages(query,context),
        temperature=0,
        max_completion_tokens=160,
        stream=True,
        include_reasoning=False,
    )
    for chunk in stream:
        content=chunk.choices[0].delta.content or ""
        if content:
            yield content

def generate_answer_stream(query:str,context:str|tuple[str,list[dict[str,Any]]]):
    ctx=context[0] if isinstance(context,tuple) else str(context)
    selected_results=context[1] if isinstance(context,tuple) and len(context)>1 else []
    if not ctx.strip():
        yield REFUSAL
        return
    if not selected_results:
        for block in re.split(r"\nSOURCE\s+\d+\n",f"\n{ctx}")[1:]:
            page_match=re.search(r"Page:\s*(\d+)",block)
            section_match=re.search(r"Section:\s*(.*)",block)
            document_id_match=re.search(r"Document ID:\s*(.*)",block)
            content_match=re.search(r"Content:\n(.*)",block,re.DOTALL)
            if content_match:
                selected_results.append({
                    "document":content_match.group(1).strip(),
                    "metadata":{
                        "document_id":document_id_match.group(1).strip() if document_id_match else "",
                        "page_number":int(page_match.group(1)) if page_match else None,
                        "section":section_match.group(1).strip() if section_match else "",
                    },
                })
    profile=query_profile(query)

    if profile["type"]=="general":
     grounded=any(
        bool(result.get("exact_evidence",False)) or
        float(result.get("evidence_match_score",0.0))>0.0 or
        float(result.get("lexical_score",0.0))>0.0 or
        (
            float(result.get("generic_relevance_score",0.0))>0.0 and
            build_generic_evidence_preview(query,"",result)
        )
        for result in selected_results
    )
    print(f"GENERATION RESULTS COUNT: {len(selected_results)}",flush=True)
    print(f"GENERATION GROUNDED: {grounded}",flush=True)
    print(f"GENERATION FIRST RESULT: {selected_results[0] if selected_results else None}",flush=True)
    if not grounded:
        print("GENERATION REJECTED: NO GROUNDING EVIDENCE",flush=True)
        return REFUSAL
    answer=generate_llm_answer(query,ctx)
    validation_result=validate_answer_grounding(answer,selected_results,query)
    print(f"GENERATION RAW ANSWER: {answer!r}",flush=True)
    print(f"GENERATION QUERY TYPE: {profile['type']}",flush=True)
    print(f"GENERATION VALIDATION: {validation_result}",flush=True)
    if not answer or answer==REFUSAL:
        return REFUSAL
    if "does not provide enough information" in normalize_text(answer):
        return REFUSAL
    if not validation_result:
        return REFUSAL
    return answer

def validate_answer_grounding(answer:str,selected_results:list[dict[str,Any]],query:str="")->bool:
    if not answer.strip() or not selected_results:
        return False
    context=normalize_text(" ".join(get_document(result) for result in selected_results))
    if not context:
        return False
    answer_numbers=extract_answer_numbers(answer)
    context_numbers=extract_numbers(context)
    if answer_numbers:
        normalized_context_numbers={normalize_number(number) for number in context_numbers}
        if not all(normalize_number(number) in normalized_context_numbers for number in answer_numbers):
            return False
    answer_terms=set(generic_answer_terms(answer))
    context_tokens={normalize_token(token) for token in tokenize(context)}
    if not answer_terms:
        return True
    is_general=query_profile(query)["type"]=="general" if query else False
    threshold=0.65 if is_general else 0.82
    supported_terms=sum(
        1 for term in answer_terms
        if normalize_token(term) in context_tokens or
        any(fuzzy_token_match(term,token)>=threshold for token in context_tokens)
    )
    support_ratio=supported_terms/max(len(answer_terms),1)
    minimum_ratio=0.25 if is_general else 0.40
    return support_ratio>=minimum_ratio

def generate_answer(query:str,context:str|tuple[str,list[dict[str,Any]]])->str:
    print(f"GENERATE_ANSWER CALLED: {query}",flush=True)
    ctx=context[0] if isinstance(context,tuple) else str(context)
    if not ctx.strip():
        print("GENERATION REJECTED: EMPTY CONTEXT",flush=True)
        return REFUSAL
    selected_results=context[1] if isinstance(context,tuple) and len(context)>1 else []
    if not selected_results:
        for block in re.split(r"\nSOURCE\s+\d+\n",f"\n{ctx}")[1:]:
            page_match=re.search(r"Page:\s*(\d+)",block)
            section_match=re.search(r"Section:\s*(.*)",block)
            document_id_match=re.search(r"Document ID:\s*(.*)",block)
            content_match=re.search(r"Content:\n(.*)",block,re.DOTALL)
            if content_match:
                selected_results.append({
                    "document":content_match.group(1).strip(),
                    "metadata":{
                        "document_id":document_id_match.group(1).strip() if document_id_match else "",
                        "page_number":int(page_match.group(1)) if page_match else None,
                        "section":section_match.group(1).strip() if section_match else "",
                    },
                })
    print(f"GENERATION RESULTS COUNT: {len(selected_results)}",flush=True)
    if not selected_results:
        print("GENERATION REJECTED: NO RETRIEVED RESULTS",flush=True)
        return REFUSAL
    profile=query_profile(query)
    print(f"GENERATION QUERY TYPE: {profile['type']}",flush=True)
    grounded=any(
        bool(result.get("exact_evidence",False)) or
        float(result.get("evidence_match_score",0.0))>0.0 or
        float(result.get("lexical_score",0.0))>0.0 or
        float(result.get("evidence_rescue_score",0.0))>0.0 or
        float(result.get("generic_relevance_score",0.0))>=0.18
        for result in selected_results
    )
    print(f"GENERATION GROUNDED: {grounded}",flush=True)
    print(f"GENERATION FIRST RESULT: {selected_results[0] if selected_results else None}",flush=True)
    if not grounded:
        print("GENERATION REJECTED: NO GROUNDING EVIDENCE",flush=True)
        return REFUSAL
    answer=generate_llm_answer(query,ctx)
    validation_result=validate_answer_grounding(answer,selected_results,query)
    print(f"GENERATION RAW ANSWER: {answer!r}",flush=True)
    print(f"GENERATION VALIDATION: {validation_result}",flush=True)
    if not answer or answer==REFUSAL:
        print("GENERATION REJECTED: EMPTY OR REFUSAL ANSWER",flush=True)
        return REFUSAL
    if "does not provide enough information" in normalize_text(answer):
        print("GENERATION REJECTED: LLM REFUSAL",flush=True)
        return REFUSAL
    if not validation_result:
        print("GENERATION REJECTED: VALIDATION FAILED",flush=True)
        return REFUSAL
    print("GENERATION ACCEPTED",flush=True)
    return answer

def format_citations(query:str,answer:str,results:list[dict[str,Any]])->str:
    if not results:
        return ""
    citations=build_citations(
        query=query,
        answer=answer,
        results=results,
        max_citations=3,
    )
    if citations:
        valid,_=validate_citations(citations,results)
        if valid:
            return format_structured_citations(citations)
    return format_generic_citations(query,answer,results,max_citations=3)

def retrieve_evidence(query:str,document_id:str|None=None)->tuple[str,list[dict[str,Any]]]:
    query=str(query).strip()
    if not query:
        return "",[]
    candidates=hybrid_retrieve(query,HYBRID_TOP_K,document_id=document_id)
    if not candidates:
        return "",[]
    if LIGHTWEIGHT_RAG:
        reranked=candidates[:RERANK_TOP_K]
    else:
        reranked=rerank_candidates(query,candidates,RERANK_TOP_K)
    protected=protect_exact_matches(query,reranked,candidates=candidates,top_k=RERANK_TOP_K)
    if document_id:
        protected=[r for r in protected if get_document_id(r)==str(document_id)]
    context,context_results=build_context(protected,query=query,top_k=CONTEXT_TOP_K)
    if document_id:
        context_results=[r for r in context_results if get_document_id(r)==str(document_id)]
    return context,context_results

def ask_question(query:str,document_id:str|None=None)->tuple[str,list[dict[str,Any]]]:
    query=str(query).strip()
    if not query:
        return REFUSAL,[]
    context,context_results=retrieve_evidence(query,document_id=document_id)
    if not context_results:
        return REFUSAL,[]
    answer=generate_answer(query,(context,context_results))
    if answer==REFUSAL:
        return answer,[]
    citations=format_citations(query,answer,context_results)
    if citations:
        answer=f"{answer.strip()}\n\n{citations}"
    return answer,context_results

def serialize_sources(results:list[dict[str,Any]])->list[dict[str,Any]]:
    serialized=[]
    seen=set()
    citation_index=0
    for result in deduplicate_results(results):
        metadata=get_metadata(result)
        key=(get_document_id(result),metadata.get("page_number"),metadata.get("section"),result.get("id"))
        if key in seen:
            continue
        seen.add(key)
        citation_index+=1
        serialized.append({
            "citation_id":f"C{citation_index}",
            "chunk_id":result.get("id") or metadata.get("chunk_id"),
            "document_id":get_document_id(result),
            "source":metadata.get("source"),
            "page_number":metadata.get("page_number"),
            "section":metadata.get("section"),
            "rrf_score":result.get("rrf_score"),
            "vector_rank":result.get("vector_rank"),
            "bm25_rank":result.get("bm25_rank"),
            "rerank_score":result.get("rerank_score"),
            "rerank_norm":result.get("rerank_norm"),
            "lexical_score":result.get("lexical_score"),
            "lexical_norm":result.get("lexical_norm"),
            "page_evidence_score":result.get("page_evidence_score"),
            "evidence_match_score":result.get("evidence_match_score"),
            "evidence_rescue_score":result.get("evidence_rescue_score"),
            "final_score":result.get("final_score"),
            "generic_relevance_score":result.get("generic_relevance_score"),
            "exact_evidence":result.get("exact_evidence",False),
        })
    return serialized

def ask_question_api(query:str,document_id:str|None=None)->RAGResponse:
    answer,sources=ask_question(query,document_id=document_id)
    return {
        "answer":answer,
        "sources":serialize_sources(sources),
        "query":query.strip(),
        "document_id":document_id,
        "source_count":len(sources),
    }

def display_sources(results:list[dict[str,Any]])->None:
    if not results:
        print("No relevant sources found.")
        return
    print("\n"+"="*120+"\nRETRIEVAL DETAILS\n"+"="*120)
    seen=set()
    for result in deduplicate_results(results):
        metadata=get_metadata(result)
        page=metadata.get("page_number","Unknown")
        section=metadata.get("section","Unknown")
        chunk_id=result.get("id","Unknown")
        key=(get_document_id(result),page,section,chunk_id)
        if key in seen:
            continue
        seen.add(key)
        print(
            f"Document: {get_document_id(result)} | Page {page} | Section: {section} | Chunk: {chunk_id} | "
            f"RRF: {float(result.get('rrf_score',0)):.4f} | "
            f"Vector: {result.get('vector_rank')} | "
            f"BM25: {result.get('bm25_rank')} | "
            f"Rerank: {float(result.get('rerank_score',0)):.4f} | "
            f"RerankNorm: {float(result.get('rerank_norm',0)):.3f} | "
            f"Lexical: {float(result.get('lexical_score',0)):.2f} | "
            f"LexNorm: {float(result.get('lexical_norm',0)):.3f} | "
            f"GenericRel: {float(result.get('generic_relevance_score',0)):.3f} | "
            f"Final: {float(result.get('final_score',0)):.4f} | "
            f"Entity: {float(result.get('entity_score',0)):.3f} | "
            f"Rel: {float(result.get('relationship_score',0)):.3f} | "
            f"PageEv: {float(result.get('page_evidence_score',0)):.3f} | "
            f"Evidence: {float(result.get('evidence_match_score',0)):.3f} | "
            f"Rescue: {float(result.get('evidence_rescue_score',0)):.2f} | "
            f"Exact: {result.get('exact_evidence',False)}"
        )

def main()->None:
    print("\nSOURVETA - PAGE-AWARE HYBRID RAG")
    while True:
        user_query=input("\nQuestion (type 'exit' to quit): ").strip()
        if user_query.lower()=="exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid question.")
            continue
        answer,sources=ask_question(user_query)
        print(f"\nANSWER:\n{answer}")
        display_sources(sources)

if __name__=="__main__":
    main()