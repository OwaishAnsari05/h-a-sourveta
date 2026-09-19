import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from generation.rag import hybrid_retrieve, retrieve_bm25, retrieve_vector

EVALUATION_FILE = "data/evaluation_questions.json"
TOP_K = 10

with open(EVALUATION_FILE, "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)


def evaluate_question(item):
    question = item["question"]
    expected_pages = set(item["expected_pages"])

    print("\n" + "=" * 70)
    print("QUESTION:\n" + question)
    print(f"EXPECTED PAGES: {expected_pages}")

    # Vector Retrieval
    vector_results = retrieve_vector(question, top_k=TOP_K)
    vector_pages = {r["metadata"]["page_number"] for r in vector_results if "metadata" in r and "page_number" in r["metadata"]}
    vector_pass = bool(expected_pages & vector_pages)

    # BM25 Retrieval
    bm25_results = retrieve_bm25(question, top_k=TOP_K)
    bm25_pages = {r["metadata"]["page_number"] for r in bm25_results if "metadata" in r and "page_number" in r["metadata"]}
    bm25_pass = bool(expected_pages & bm25_pages)

    # Hybrid Retrieval
    hybrid_results = hybrid_retrieve(question)
    hybrid_pages = {r["metadata"]["page_number"] for r in hybrid_results if "metadata" in r and "page_number" in r["metadata"]}
    hybrid_pass = bool(expected_pages & hybrid_pages)

    # Display Results
    print(f"\nVECTOR TOP-{TOP_K}: {sorted(vector_pages)} | Result: {'PASS' if vector_pass else 'FAIL'}")
    print(f"BM25 TOP-{TOP_K}  : {sorted(bm25_pages)} | Result: {'PASS' if bm25_pass else 'FAIL'}")
    print(f"HYBRID TOP-{TOP_K}: {sorted(hybrid_pages)} | Result: {'PASS' if hybrid_pass else 'FAIL'}")

    return {"vector": vector_pass, "bm25": bm25_pass, "hybrid": hybrid_pass}


results = [evaluate_question(item) for item in evaluation_data]
total = len(results)

if total == 0:
    print("No evaluation questions found.")
    sys.exit()

vector_recall = sum(r["vector"] for r in results) / total
bm25_recall = sum(r["bm25"] for r in results) / total
hybrid_recall = sum(r["hybrid"] for r in results) / total

print("\n" + "=" * 70)
print("RETRIEVAL METHOD COMPARISON")
print("=" * 70)
print(f"Total Questions  : {total}")
print(f"Vector Recall@{TOP_K} : {vector_recall:.2%}")
print(f"BM25 Recall@{TOP_K}   : {bm25_recall:.2%}")
print(f"Hybrid Recall@{TOP_K} : {hybrid_recall:.2%}")
print("=" * 70)