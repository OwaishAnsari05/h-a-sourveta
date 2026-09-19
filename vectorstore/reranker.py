import re
from sentence_transformers import CrossEncoder

# CONFIGURATION & INITIALIZATION

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("Loading reranker model...")
reranker = CrossEncoder(RERANKER_MODEL)
print("Reranker loaded successfully.")

# TEXT & METADATA HELPERS

def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def get_document(result):
    if isinstance(result, dict):
        return str(result.get("document", "")).strip()
    return str(result).strip()


def get_metadata(result):
    if isinstance(result, dict):
        return result.get("metadata", {}) or {}
    return {}


def build_reranker_text(result):
    """Include metadata context along with document content."""
    meta = get_metadata(result)
    return (
        f"Section: {meta.get('section', '')}\n"
        f"Source: {meta.get('source', '')}\n"
        f"Page: {meta.get('page_number', '')}\n"
        f"Content: {get_document(result)}"
    )

# LEXICAL EVIDENCE SCORING

def lexical_score(query, result):
    """Calculate a bounded lexical boost for exact financial/entity terms."""
    q = normalize_text(query)
    document = normalize_text(get_document(result))
    section = normalize_text(get_metadata(result).get("section", ""))
    searchable = f"{section} {document}"

    exact_phrases = {
        "joint venture": 8.0,
        "joint ventures": 8.0,
        "venture partners": 8.0,
        "cash and cash equivalents": 8.0,
        "bank balance other than cash and cash equivalents": 12.0,
        "bank deposits": 6.0,
        "provision for standard assets": 12.0,
        "total exposure to top five npa accounts": 12.0,
        "total income": 10.0,
        "total expenditure": 10.0,
    }

    score = sum(weight for phrase, weight in exact_phrases.items() if phrase in q and phrase in searchable)

    if "joint venture" in q or "venture partners" in q:
        if "tata sons private limited" in searchable:
            score += 10.0
        if "tata chemicals limited" in searchable:
            score += 10.0

    return score

# RERANK DOCUMENTS

def rerank_documents(query, results, top_k=10):
    """
    Rerank dictionary results or plain strings.
    - Dict input: returns dicts with updated scores.
    - String input: returns (document, score) tuples.
    """
    if not results:
        return []

    is_dict_input = isinstance(results[0], dict)
    pairs = [[query, build_reranker_text(result)] for result in results]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = []
    for result, score in zip(results, scores):
        score_val = float(score)

        if not isinstance(result, dict):
            ranked.append((get_document(result), score_val))
            continue

        item = result.copy()
        item["rerank_score"] = score_val
        item["reranker_lexical_score"] = lexical_score(query, item)
        item["combined_rank_score"] = score_val + min(item["reranker_lexical_score"] * 0.35, 5.0)
        ranked.append(item)

    if is_dict_input:
        ranked.sort(key=lambda x: x["combined_rank_score"], reverse=True)
        return ranked[:top_k]

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]