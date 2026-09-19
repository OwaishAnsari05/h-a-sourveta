import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generation.rag import (
    deduplicate_results,
    get_page,
    hybrid_retrieve,
    page_aware_rerank,
    protect_exact_matches,
)

HYBRID_TOP_K = 30
RERANK_TOP_K = 10

EVALUATION_QUESTIONS = [
    {
        "question": "What was the total income in FY 2024-25?",
        "expected_pages": {5},
        "expected_content": ["31,455.43", "37,569.16", "lakhs"],
    },
    {
        "question": "What was the total expenditure in FY 2024-25?",
        "expected_pages": {5},
        "expected_content": ["27,897.66", "57,566.17", "lakhs"],
    },
    {
        "question": "What was the cash and cash equivalents as of March 31, 2025?",
        "expected_pages": {151},
        "expected_content": ["1,819.57", "lakhs"],
    },
    {
        "question": "What was the bank balance other than cash and cash equivalents as of March 31, 2025?",
        "expected_pages": {151},
        "expected_content": ["52.63", "lakhs"],
    },
    {
        "question": "What was the provision for standard assets as of March 31, 2025?",
        "expected_pages": {101},
        "expected_content": ["Provision for Standard Assets"],
    },
    {
        "question": "Who are the joint venture partners?",
        "expected_pages": {101},
        "expected_content": ["Tata Sons Private Limited", "Tata Chemicals Limited"],
    },
    {
        "question": "What was the total exposure to the top five NPA accounts?",
        "expected_pages": {101},
        "expected_content": ["NIL"],
    },
]


def get_pages(results):
    if not results:
        return []
    return [get_page(r) for r in results if get_page(r) is not None]


def get_expected_rank(results, expected_pages):
    for rank, result in enumerate(results, 1):
        if get_page(result) in expected_pages:
            return rank
    return None


def page_recall(results, expected_pages):
    return bool(set(get_pages(results)).intersection(expected_pages))


def reciprocal_rank(results, expected_pages):
    rank = get_expected_rank(results, expected_pages)
    return 1.0 / rank if rank else 0.0


def has_exact_evidence(results):
    return any(r.get("exact_evidence", False) for r in results)


def exact_evidence_rank(results):
    for rank, result in enumerate(results, 1):
        if result.get("exact_evidence", False):
            return rank
    return None


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_results(label, results):
    print(f"\n{label}")
    print("-" * 70)
    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, 1):
        metadata = result.get("metadata", {}) or {}
        page = metadata.get("page_number", "Unknown")
        section = metadata.get("section", "Unknown")
        rerank_score = safe_float(result.get("rerank_score"))
        lexical_score = safe_float(result.get("lexical_score"))
        final_score = safe_float(result.get("final_score"))
        exact = result.get("exact_evidence", False)

        print(
            f"{rank:2d}. Page {page:<4} | "
            f"Final {final_score:8.4f} | "
            f"Rerank {rerank_score:8.4f} | "
            f"Lexical {lexical_score:6.2f} | "
            f"Exact {str(exact):5} | "
            f"{section}"
        )


def evaluate_question(number, item):
    question = item["question"]
    expected_pages = item["expected_pages"]

    print("\n" + "=" * 70)
    print(f"QUESTION {number}/{len(EVALUATION_QUESTIONS)}")
    print("=" * 70)
    print(f"QUESTION: {question}")
    print(f"EXPECTED PAGE(S): {sorted(expected_pages)}")

    # 1. Hybrid Retrieval
    print("\nSTEP 1: HYBRID RETRIEVAL\n" + "-" * 70)
    hybrid_results = deduplicate_results(hybrid_retrieve(question, top_k=HYBRID_TOP_K))[:HYBRID_TOP_K]
    hybrid_rank = get_expected_rank(hybrid_results, expected_pages)
    hybrid_rr = reciprocal_rank(hybrid_results, expected_pages)
    hybrid_pass = page_recall(hybrid_results, expected_pages)

    print(f"Retrieved candidates : {len(hybrid_results)}")
    print(f"Pages                : {get_pages(hybrid_results)}")
    print(f"Expected page rank   : {hybrid_rank or 'NOT FOUND'}")
    print(f"Recall@{HYBRID_TOP_K:<13}: {'PASS' if hybrid_pass else 'FAIL'}")
    print_results("HYBRID TOP RESULTS", hybrid_results[:10])

    # 2. Page-Aware Reranking
    print("\nSTEP 2: PAGE-AWARE RERANKING\n" + "-" * 70)
    reranked_results = deduplicate_results(
        page_aware_rerank(question, hybrid_results, top_k=RERANK_TOP_K)
    )[:RERANK_TOP_K]
    rerank_rank = get_expected_rank(reranked_results, expected_pages)
    rerank_rr = reciprocal_rank(reranked_results, expected_pages)
    rerank_pass = page_recall(reranked_results, expected_pages)

    print(f"Reranked results      : {len(reranked_results)}")
    print(f"Pages                 : {get_pages(reranked_results)}")
    print(f"Expected page rank    : {rerank_rank or 'NOT FOUND'}")
    print(f"Recall@{RERANK_TOP_K:<13}: {'PASS' if rerank_pass else 'FAIL'}")
    print_results("PAGE-AWARE RERANK RESULTS", reranked_results)

    # 3. Exact-Evidence Protection
    print("\nSTEP 3: EXACT-EVIDENCE PROTECTION\n" + "-" * 70)
    protected_results = deduplicate_results(
        protect_exact_matches(
            question,
            reranked_results,
            candidates=hybrid_results,
            top_k=RERANK_TOP_K,
        )
    )[:RERANK_TOP_K]

    protected_rank = get_expected_rank(protected_results, expected_pages)
    protected_rr = reciprocal_rank(protected_results, expected_pages)
    protected_pass = page_recall(protected_results, expected_pages)
    exact_rank = exact_evidence_rank(protected_results)
    exact_found = has_exact_evidence(protected_results)

    print(f"Protected results     : {len(protected_results)}")
    print(f"Pages                 : {get_pages(protected_results)}")
    print(f"Expected page rank    : {protected_rank or 'NOT FOUND'}")
    print(f"Recall@{RERANK_TOP_K:<13}: {'PASS' if protected_pass else 'FAIL'}")
    print(f"Exact evidence        : {'FOUND' if exact_found else 'NOT FOUND'}")
    print(f"Exact evidence rank   : {exact_rank or 'NOT FOUND'}")
    print_results("PROTECTED FINAL RESULTS", protected_results)

    return {
        "hybrid_pass": hybrid_pass,
        "rerank_pass": rerank_pass,
        "protected_pass": protected_pass,
        "hybrid_rank": hybrid_rank,
        "rerank_rank": rerank_rank,
        "protected_rank": protected_rank,
        "hybrid_rr": hybrid_rr,
        "rerank_rr": rerank_rr,
        "protected_rr": protected_rr,
        "exact_found": exact_found,
        "exact_rank": exact_rank,
    }


def main():
    total = len(EVALUATION_QUESTIONS)
    if total == 0:
        print("No evaluation questions found.")
        sys.exit(1)

    print("=" * 70 + "\nTATA ANNUAL REPORT - RERANKER EVALUATION\n" + "=" * 70)
    print(f"Total Questions : {total}")
    print(f"Hybrid Top-K    : {HYBRID_TOP_K}")
    print(f"Rerank Top-K    : {RERANK_TOP_K}")
    print("=" * 70)

    results = []
    for number, item in enumerate(EVALUATION_QUESTIONS, 1):
        try:
            results.append(evaluate_question(number, item))
        except Exception as exc:
            print(f"\nEVALUATION ERROR\nQuestion: {item['question']}\nError   : {exc}")
            results.append({
                "hybrid_pass": False,
                "rerank_pass": False,
                "protected_pass": False,
                "hybrid_rank": None,
                "rerank_rank": None,
                "protected_rank": None,
                "hybrid_rr": 0.0,
                "rerank_rr": 0.0,
                "protected_rr": 0.0,
                "exact_found": False,
                "exact_rank": None,
            })

    hybrid_recall = sum(r["hybrid_pass"] for r in results) / total
    rerank_recall = sum(r["rerank_pass"] for r in results) / total
    protected_recall = sum(r["protected_pass"] for r in results) / total

    hybrid_mrr = sum(r["hybrid_rr"] for r in results) / total
    rerank_mrr = sum(r["rerank_rr"] for r in results) / total
    protected_mrr = sum(r["protected_rr"] for r in results) / total

    exact_count = sum(r["exact_found"] for r in results)
    rank_improved = sum(
        1 for r in results
        if r["hybrid_rank"] is not None and r["protected_rank"] is not None and r["protected_rank"] < r["hybrid_rank"]
    )
    rank_same = sum(
        1 for r in results
        if r["hybrid_rank"] is not None and r["protected_rank"] is not None and r["protected_rank"] == r["hybrid_rank"]
    )

    print("\n\n" + "=" * 70 + "\nRERANKER EVALUATION SUMMARY\n" + "=" * 70)
    print(f"Total Questions          : {total}")
    print(f"Hybrid Recall@{HYBRID_TOP_K:<2}       : {hybrid_recall:.2%}")
    print(f"Reranker Recall@{RERANK_TOP_K:<2}     : {rerank_recall:.2%}")
    print(f"Protected Recall@{RERANK_TOP_K:<2}    : {protected_recall:.2%}")
    print(f"Hybrid MRR               : {hybrid_mrr:.4f}")
    print(f"Reranker MRR             : {rerank_mrr:.4f}")
    print(f"Protected MRR            : {protected_mrr:.4f}")
    print(f"Exact Evidence Coverage  : {exact_count}/{total} ({exact_count / total:.2%})")
    print(f"Rank Improved            : {rank_improved}/{total}")
    print(f"Rank Unchanged           : {rank_same}/{total}")
    print("=" * 70)

    print("\nQUESTION-BY-QUESTION RANKING\n" + "-" * 70)
    for number, (item, result) in enumerate(zip(EVALUATION_QUESTIONS, results), 1):
        print(
            f"{number}. {item['question']}\n"
            f"   Hybrid: {result['hybrid_rank'] or 'N/A'} -> "
            f"Rerank: {result['rerank_rank'] or 'N/A'} -> "
            f"Protected: {result['protected_rank'] or 'N/A'} | "
            f"Exact: {result['exact_rank'] or 'N/A'}"
        )

    print("\n" + "=" * 70)
    if protected_recall == 1.0 and protected_mrr >= hybrid_mrr:
        print("STATUS: PASS\nReranking and exact-evidence protection preserved full page coverage.")
    else:
        print("STATUS: REVIEW\nOne or more reranking metrics require investigation.")
    print("=" * 70)


if __name__ == "__main__":
    main()