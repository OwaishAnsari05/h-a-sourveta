import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT & IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generation.citation import build_citations, validate_citations

# ============================================================
# TEST DATA HELPER
# ============================================================

def make_result(text, page_number, section, section_index, chunk_index):
    chunk_id = f"tata_annual_report_2024_25_p{page_number}_s{section_index}_c{chunk_index}"
    return {
        "id": chunk_id,
        "document": text,
        "metadata": {
            "chunk_id": chunk_id,
            "document_id": "tata_annual_report_2024_25",
            "source": "Tata Annual Report 2024-25",
            "page_number": page_number,
            "section": section,
            "section_index": section_index,
            "chunk_index": chunk_index,
        },
    }

# ============================================================
# UNIT TESTS
# ============================================================

def test_total_income_citation():
    result = make_result(
        "Total Income | 31,455.43 | 24,796.96 | 37,569.16 | 31,813.11 lakhs",
        5, "Financial Results", 2, 1
    )
    answer = "The total income for FY 2024-25 was 31,455.43 lakhs (standalone) and 37,569.16 lakhs (consolidated)."
    citations = build_citations(query="What was the total income in FY 2024-25?", answer=answer, results=[result])

    assert len(citations) == 1
    assert citations[0].citation_id == "C1"
    assert citations[0].page_number == 5
    assert citations[0].chunk_id.endswith("_p5_s2_c1")
    assert "31,455.43" in citations[0].evidence


def test_cash_citation():
    cash_result = make_result(
        "Cash on hand - Balances with banks | 1,515.88 | Other bank balances | 303.69 | 1,819.57 lakhs",
        151, "Cash and cash equivalents", 4, 1
    )
    neighboring_result = make_result(
        "Bank balance other than cash and cash equivalents | 52.63 lakhs",
        151, "Bank balance other than cash", 5, 1
    )
    answer = "The cash and cash equivalents were 1,819.57 lakhs."
    citations = build_citations(
        query="What was the cash and cash equivalents as of March 31, 2025?",
        answer=answer,
        results=[neighboring_result, cash_result],
    )

    assert len(citations) >= 1
    assert citations[0].page_number == 151
    assert citations[0].chunk_id.endswith("_p151_s4_c1")


def test_bank_balance_citation():
    result = make_result(
        "Bank balance other than cash and cash equivalents | 52.63 lakhs",
        151, "Bank balance other than cash and cash equivalents", 5, 1
    )
    answer = "The bank balance other than cash and cash equivalents was 52.63 lakhs."
    citations = build_citations(
        query="What was the bank balance other than cash and cash equivalents as of March 31, 2025?",
        answer=answer,
        results=[result],
    )

    assert len(citations) == 1
    assert citations[0].page_number == 151
    assert "52.63" in citations[0].evidence


def test_joint_venture_citation():
    result = make_result(
        "Joint Venture Partners: Tata Sons Private Limited and Tata Chemicals Limited",
        101, "Joint Venture Partners", 1, 1
    )
    answer = "The joint venture partners are Tata Sons Private Limited and Tata Chemicals Limited."
    citations = build_citations(query="Who are the joint venture partners?", answer=answer, results=[result])

    assert len(citations) == 1
    assert citations[0].page_number == 101
    assert "Tata Sons Private Limited" in citations[0].evidence


def test_provision_citation():
    result = make_result(
        "Provision for Standard Assets | 1.07 | (4.67) lakhs",
        101, "Provisions and Contingencies", 6, 1
    )
    answer = "The provision for standard assets was 1.07 lakhs."
    citations = build_citations(
        query="What was the provision for standard assets as of March 31, 2025?",
        answer=answer,
        results=[result],
    )

    assert len(citations) == 1
    assert citations[0].page_number == 101
    assert "1.07" in citations[0].evidence


def test_npa_citation():
    result = make_result(
        "Total exposure to the top five NPA accounts: NIL",
        101, "Non Performing Assets", 7, 1
    )
    answer = "The total exposure to the top five NPA accounts was NIL."
    citations = build_citations(
        query="What was the total exposure to the top five NPA accounts?",
        answer=answer,
        results=[result],
    )

    assert len(citations) == 1
    assert citations[0].page_number == 101
    assert "NIL" in citations[0].evidence


def test_citation_validation():
    result = make_result("Provision for Standard Assets | 1.07 lakhs", 101, "Provisions and Contingencies", 6, 1)
    answer = "The provision for standard assets was 1.07 lakhs."
    citations = build_citations(query="What was the provision for standard assets?", answer=answer, results=[result])

    valid, errors = validate_citations(citations, [result])
    assert valid, errors


def test_unsupported_answer_has_no_citation():
    result = make_result("This chunk contains unrelated information about another topic.", 200, "Unrelated", 1, 1)
    answer = "The provision for standard assets was 1.07 lakhs."
    citations = build_citations(query="What was the provision for standard assets?", answer=answer, results=[result])

    assert citations == []