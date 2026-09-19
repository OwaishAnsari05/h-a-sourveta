from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class Citation:
    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    page_number: Optional[int]
    section: str
    section_index: Optional[int]
    chunk_index: Optional[int]
    evidence: str
    support_score: float


def get_metadata(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        metadata = result.get("metadata")
        return metadata if isinstance(metadata, dict) else result
    metadata = getattr(result, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def get_chunk_id(result: Any) -> str:
    if isinstance(result,dict):
        return str(result.get("chunk_id") or result.get("id") or result.get("metadata",{}).get("chunk_id") or "")
    return str(getattr(result,"chunk_id",None) or getattr(result,"id",None) or "")


def get_text(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("text") or result.get("document") or result.get("content") or "")
    return str(getattr(result, "text", None) or getattr(result, "document", None) or getattr(result, "content", None) or "")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).lower().strip()


def extract_numbers(text: str) -> List[str]:
    return re.findall(
        r"(?<![\w.])(?:₹\s*)?\d[\d,]*(?:\.\d+)?(?:\s*%|(?:\s*(?:lakhs?|crores?|million|billion)))?",
        str(text or ""),
        flags=re.I,
    )


def extract_answer_numbers(answer: str) -> List[str]:
    numbers = extract_numbers(answer)
    cleaned = []
    for value in numbers:
        raw = value.strip()
        numeric = re.sub(r"[₹,%\s]", "", raw)
        numeric = re.sub(r"(?i)(lakhs?|crores?|million|billion)$", "", numeric).strip().replace(",", "")
        if not numeric:
            continue
        try:
            number = float(numeric)
        except ValueError:
            continue
        if number.is_integer() and 1 <= number <= 31:
            continue
        if number.is_integer() and 1900 <= number <= 2100:
            continue
        cleaned.append(raw)
    return cleaned


MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


def is_year(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}", value.strip()))


def is_date_like(value: str) -> bool:
    value = value.strip().lower()
    return bool(re.search(rf"\b\d{{1,2}}[\s/-]+{MONTHS}[\s,/-]+\d{{4}}\b", value)) or bool(re.search(rf"\b{MONTHS}[\s-]+\d{{1,2}}[\s,/-]+\d{{4}}\b", value))


def extract_answer_phrases(answer: str) -> List[str]:
    phrases = []
    for match in re.findall(r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*){1,7}\b", answer):
        phrase = match.strip()
        if len(phrase) < 4 or is_year(phrase):
            continue
        phrases.append(phrase)
    return list(dict.fromkeys(phrases))


KNOWN_ENTITIES = [
    "tata sons private limited",
    "tata chemicals limited",
    "cash and cash equivalents",
    "bank balance other than cash and cash equivalents",
    "provision for standard assets",
    "total income",
    "total expenditure",
    "total exposure to the top five npa accounts",
]


def extract_answer_entities(answer: str) -> List[str]:
    normalized = normalize_text(answer)
    entities = []
    for entity in KNOWN_ENTITIES:
        if entity in normalized:
            entities.append(entity)
    for phrase in extract_answer_phrases(answer):
        phrase = normalize_text(phrase)
        if len(phrase.split()) >= 2 and phrase not in entities:
            entities.append(phrase)
    return list(dict.fromkeys(entities))


def find_matching_evidence(answer: str, chunk_text: str) -> Dict[str, Any]:
    answer_numbers = extract_answer_numbers(answer)
    answer_entities = extract_answer_entities(answer)
    normalized_chunk = normalize_text(chunk_text)
    matched_numbers = []
    matched_entities = []

    for number in answer_numbers:
        numeric = normalize_number(number)
        if numeric and numeric in re.sub(r"[₹,%\s]", "", normalized_chunk):
            matched_numbers.append(number)

    for entity in answer_entities:
        if normalize_text(entity) in normalized_chunk:
            matched_entities.append(entity)

    return {
        "numbers": list(dict.fromkeys(matched_numbers)),
        "entities": list(dict.fromkeys(matched_entities)),
        "exact": bool(matched_numbers or matched_entities),
    }


def query_overlap_score(query: str, chunk_text: str) -> float:
    query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", normalize_text(query)))
    chunk_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", normalize_text(chunk_text)))
    if not query_words:
        return 0.0
    return len(query_words & chunk_words) / len(query_words)


def calculate_support_score(answer: str, query: str, result: Any) -> Tuple[float, Dict[str, Any]]:
    text = get_text(result)
    evidence = find_matching_evidence(answer, text)
    score = len(evidence["numbers"]) * 1000.0
    score += len(evidence["entities"]) * 100.0
    if evidence["exact"]:
        score += 500.0
    score += query_overlap_score(query, text) * 0.001
    return score, evidence


def normalize_number(value: str) -> str:
    numeric = re.sub(r"[₹,%\s]", "", str(value or ""))
    numeric = re.sub(r"(?i)(lakhs?|crores?|million|billion)$", "", numeric).strip()
    return numeric.replace(",", "")


def line_matches_number(line: str, number: str) -> bool:
    numeric = normalize_number(number)
    if not numeric:
        return False
    normalized_line = re.sub(r"[₹,%\s]", "", normalize_text(line))
    return numeric in normalized_line


def line_matches_entity(line: str, entity: str) -> bool:
    return normalize_text(entity) in normalize_text(line)


def find_label_line(lines: List[str], answer_entities: List[str]) -> Optional[int]:
    if not answer_entities:
        return None
    priority = sorted(answer_entities,key=lambda entity: len(entity.split()),reverse=True)
    for entity in priority:
        normalized_entity = normalize_text(entity)
        for index, line in enumerate(lines):
            if normalized_entity in normalize_text(line):
                return index
    return None


def is_numeric_line(line: str) -> bool:
    return bool(re.search(r"\d", line))

def build_table_row_evidence(answer: str, text: str, max_chars: int = 1000) -> str:
    answer_entities = extract_answer_entities(answer)
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    financial_entities = {
        "total income",
        "total expenditure",
        "cash and cash equivalents",
        "bank balance other than cash and cash equivalents",
        "provision for standard assets",
        "total exposure to the top five npa accounts",
    }
    entities = [e for e in answer_entities if e in financial_entities]
    if not entities:
        return ""

    label_index = find_label_line(lines,entities)
    if label_index is None:
        return ""

    label = normalize_text(lines[label_index])
    entity = next((e for e in entities if normalize_text(e) in label),None)
    if entity is None:
        return ""

    numeric_lines = []
    for line in lines[label_index + 1:]:
        if normalize_text(line) in {normalize_text("total expenditure"),normalize_text("profit / (loss) before tax from continuing operations, exceptional items and share of profit of equity accounted investees and income tax")}:
            break
        if re.fullmatch(r"\(?\d[\d,]*(?:\.\d+)?\)?",line):
            numeric_lines.append(line)
        elif line == "-":
            numeric_lines.append(line)
        if len(numeric_lines) >= 4:
            break

    if entity in {"total income","total expenditure"} and len(numeric_lines) >= 4:
        return " | ".join([lines[label_index]] + numeric_lines[:4])[:max_chars]

    answer_numbers = extract_answer_numbers(answer)
    matched = []
    for line in lines[label_index + 1:label_index + 8]:
        if any(normalize_number(number) == normalize_number(line) for number in answer_numbers):
            matched.append(line)
    if matched:
        return " | ".join([lines[label_index]] + matched)[:max_chars]

    return lines[label_index][:max_chars]


def build_entity_evidence(answer: str, text: str, max_chars: int = 1000) -> str:
    answer_entities = extract_answer_entities(answer)
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    matched_indices = []
    for index,line in enumerate(lines):
        if any(line_matches_entity(line,entity) for entity in answer_entities):
            matched_indices.append(index)

    if not matched_indices:
        return ""

    evidence_lines = []
    for index in matched_indices:
        start = max(0,index - 1)
        end = min(len(lines),index + 2)
        for line in lines[start:end]:
            if line not in evidence_lines:
                evidence_lines.append(line)

    return " | ".join(evidence_lines)[:max_chars]


def build_evidence_preview(answer: str, text: str, max_chars: int = 1000) -> str:
    table_evidence = build_table_row_evidence(answer,text,max_chars)
    if table_evidence:
        return table_evidence

    entity_evidence = build_entity_evidence(answer,text,max_chars)
    if entity_evidence:
        return entity_evidence

    answer_numbers = extract_answer_numbers(answer)
    answer_entities = extract_answer_entities(answer)
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]

    if not lines:
        return ""

    evidence_lines = []
    for line in lines:
        normalized_line = normalize_text(line)
        matched = any(line_matches_number(line,number) for number in answer_numbers)
        if not matched:
            matched = any(normalize_text(entity) in normalized_line for entity in answer_entities)
        if matched and line not in evidence_lines:
            evidence_lines.append(line)

    if evidence_lines:
        return " | ".join(evidence_lines)[:max_chars]

    return re.sub(r"\s+"," ",str(text or "")).strip()[:max_chars]


def build_citations(
    answer: str,
    query: str,
    results: Sequence[Any],
    max_citations: int = 5,
) -> List[Citation]:
    scored = []

    for result in results:
        score,evidence = calculate_support_score(answer,query,result)
        if evidence["exact"]:
            scored.append((score,result,evidence))

    scored.sort(key=lambda item:item[0],reverse=True)

    answer_numbers = extract_answer_numbers(answer)
    answer_entities = extract_answer_entities(answer)
    covered_numbers = set()
    covered_entities = set()
    citation_keys = set()
    citations = []

    for score,result,evidence in scored:
        matched_numbers = set(evidence["numbers"])
        matched_entities = set(evidence["entities"])
        new_numbers = matched_numbers - covered_numbers
        new_entities = matched_entities - covered_entities

        metadata = get_metadata(result)
        page = metadata.get("page_number",metadata.get("page"))

        try:
            page = int(page) if page is not None else None
        except (TypeError,ValueError):
            page = None

        document_id = str(metadata.get("document_id") or metadata.get("document") or "")
        section = str(metadata.get("section") or metadata.get("section_title") or "")
        citation_key = (document_id,page,normalize_text(section))

        if citation_key in citation_keys:
            covered_numbers.update(matched_numbers)
            covered_entities.update(matched_entities)
            continue

        if citations and not new_numbers and not new_entities:
            continue

        citations.append(
            Citation(
                citation_id=f"C{len(citations) + 1}",
                chunk_id=get_chunk_id(result),
                document_id=document_id,
                source=str(metadata.get("source") or metadata.get("filename") or metadata.get("file_name") or ""),
                page_number=page,
                section=section,
                section_index=metadata.get("section_index"),
                chunk_index=metadata.get("chunk_index"),
                evidence=build_evidence_preview(answer,get_text(result)),
                support_score=score,
            )
        )

        citation_keys.add(citation_key)
        covered_numbers.update(matched_numbers)
        covered_entities.update(matched_entities)

        numbers_complete = not answer_numbers or set(answer_numbers).issubset(covered_numbers)
        entities_complete = not answer_entities or set(answer_entities).issubset(covered_entities)

        if numbers_complete and entities_complete:
            break

        if len(citations) >= max_citations:
            break

    return citations


def validate_citations(
    citations: Sequence[Citation],
    results: Sequence[Any],
) -> Tuple[bool,List[str]]:
    valid_chunk_ids = {get_chunk_id(result) for result in results}
    errors = []

    for citation in citations:
        if not citation.chunk_id:
            errors.append(f"{citation.citation_id}: missing chunk_id")
        elif citation.chunk_id not in valid_chunk_ids:
            errors.append(f"{citation.citation_id}: chunk_id not found in retrieved results")

        if citation.page_number is None:
            errors.append(f"{citation.citation_id}: missing page number")

        if not citation.evidence.strip():
            errors.append(f"{citation.citation_id}: missing evidence")

    return len(errors) == 0,errors


def format_citations(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""

    lines = []
    for citation in citations:
        location = f"Page {citation.page_number}" if citation.page_number is not None else "Page ?"
        if citation.section:
            location += f", {citation.section}"
        lines.append(f"[{citation.citation_id}] {location} — {citation.evidence}")

    return "\n".join(lines)


def citations_to_dicts(
    citations: Iterable[Citation],
) -> List[Dict[str,Any]]:
    return [
        {
            "citation_id": citation.citation_id,
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "source": citation.source,
            "page_number": citation.page_number,
            "section": citation.section,
            "section_index": citation.section_index,
            "chunk_index": citation.chunk_index,
            "evidence": citation.evidence,
            "support_score": citation.support_score,
        }
        for citation in citations
    ]


if __name__ == "__main__":
    results = [
        {
            "text": "1. Joint Venture Partners\nThe joint venture partners are Tata Sons Private Limited and Tata Chemicals Limited.",
            "metadata": {
                "chunk_id": "tata_annual_report_2024_25_p101_s1_c1",
                "document_id": "tata_annual_report_2024_25",
                "source": "tata_annual_report_2024_25.pdf",
                "page_number": 101,
                "section": "1. Joint Venture Partners",
                "section_index": 1,
                "chunk_index": 1,
            },
        },
        {
            "text": "As at March 31, 2025, the balance was reported.",
            "metadata": {
                "chunk_id": "tata_annual_report_2024_25_p151_date_c1",
                "document_id": "tata_annual_report_2024_25",
                "source": "tata_annual_report_2024_25.pdf",
                "page_number": 151,
                "section": "Notes",
                "section_index": 2,
                "chunk_index": 1,
            },
        },
    ]

    query = "Who are the joint venture partners?"
    answer = "The joint venture partners are Tata Sons Private Limited and Tata Chemicals Limited."

    citations = build_citations(answer=query if False else answer,query=query,results=results)
    valid,errors = validate_citations(citations,results)

    print("=" * 80)
    print("CITATION TEST")
    print("=" * 80)

    for citation in citations:
        print(f"{citation.citation_id} | Page {citation.page_number} | Score {citation.support_score:.3f}")
        print(f"Chunk: {citation.chunk_id}")
        print(f"Evidence: {citation.evidence}")
        print("-" * 80)

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print(f"Valid: {valid}")

    if errors:
        for error in errors:
            print(error)
    else:
        print("No validation errors.")

    print()
    print("=" * 80)
    print("EXPECTED")
    print("=" * 80)
    print("C1 -> Page 101")
    print("Evidence should contain both:")
    print("     Tata Sons Private Limited")
    print("     Tata Chemicals Limited")
    print("Date-only chunks -> NOT CITED")