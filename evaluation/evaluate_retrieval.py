import json
import os
import sys

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0,PROJECT_ROOT)

from generation.rag import hybrid_retrieve

EVALUATION_FILE=os.path.join(PROJECT_ROOT,"data","evaluation_questions.json")
TOP_K=10

with open(EVALUATION_FILE,"r",encoding="utf-8") as file:
    evaluation_data=json.load(file)

print("="*70)
print("RETRIEVAL EVALUATION")
print("="*70)
print(f"Total evaluation questions: {len(evaluation_data)}")
print(f"Evaluation cutoff: Top-{TOP_K}")

def get_pages(results,top_k=TOP_K):
    pages=set()
    for result in (results or [])[:top_k]:
        metadata=result.get("metadata",{}) or {}
        page=metadata.get("page_number")
        if page is not None:
            try:
                pages.add(int(page))
            except (ValueError,TypeError):
                continue
    return pages

def evaluate_question(item):
    question=item["question"]
    expected_pages=set(item["expected_pages"])

    print("\n"+"="*70)
    print(f"QUESTION:\n{question}")
    print(f"EXPECTED PAGES: {sorted(expected_pages)}")

    hybrid_results=hybrid_retrieve(question,top_k=TOP_K)
    hybrid_pages=get_pages(hybrid_results)
    hybrid_pass=bool(expected_pages&hybrid_pages)

    print(f"\nHYBRID TOP-{TOP_K}: {sorted(hybrid_pages)}")
    print(f"Result: {'PASS' if hybrid_pass else 'FAIL'}")
    print(f"Retrieved results: {len(hybrid_results)}")

    if hybrid_results:
        print("\nTOP RESULTS:")
        for index,result in enumerate(hybrid_results[:TOP_K],start=1):
            metadata=result.get("metadata",{}) or {}
            page=metadata.get("page_number","Unknown")
            section=metadata.get("section","Unknown")
            chunk_id=result.get("id","")
            print(f"{index}. Page {page} | Section: {section} | Chunk ID: {chunk_id}")

    return {"hybrid":hybrid_pass}

results=[evaluate_question(item) for item in evaluation_data]
total=len(results)

if total==0:
    print("\nNo evaluation questions found.")
    sys.exit(1)

hybrid_recall=sum(result["hybrid"] for result in results)/total

print("\n"+"="*70)
print("RETRIEVAL EVALUATION SUMMARY")
print("="*70)
print(f"Total Questions   : {total}")
print(f"Hybrid Recall@{TOP_K} : {hybrid_recall:.2%}")
print("="*70)

if hybrid_recall==1.0:
    print("STATUS: PASS")
    print("Hybrid retrieval found the expected page for every evaluation question.")
else:
    print("STATUS: REVIEW")
    print("Hybrid retrieval did not find the expected page for every evaluation question.")

print("="*70)