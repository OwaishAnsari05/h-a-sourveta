import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from generation.rag import hybrid_retrieve,build_context,generate_answer,validate_answer_grounding

QUESTIONS_FILE=ROOT/"data"/"evaluation_questions.json"
RETRIEVAL_TOP_K=15
CONTEXT_TOP_K=5

def load_questions():
    with open(QUESTIONS_FILE,"r",encoding="utf-8") as f:
        data=json.load(f)
    if isinstance(data,dict):
        for key in ("questions","evaluation_questions","data"):
            if key in data:
                data=data[key]
                break
    return data

def get_question(item):
    for key in ("question","query","prompt"):
        if key in item:
            return str(item[key]).strip()
    return ""

def get_expected_pages(item):
    for key in ("expected_page","page","expected_pages"):
        if key in item:
            value=item[key]
            return value if isinstance(value,list) else [value]
    return []

def get_expected_answer(item):
    for key in ("expected_answer","answer","reference_answer","ground_truth"):
        if key in item and item[key] is not None:
            return str(item[key]).strip()
    return ""

def get_page(result):
    metadata=result.get("metadata",{}) or {}
    page=metadata.get("page_number",metadata.get("page"))
    if page is None:
        page=result.get("page_number",result.get("page"))
    try:
        return int(page)
    except (TypeError,ValueError):
        return page

def normalize_text(text):
    text=str(text or "").lower()
    text=text.replace("₹","")
    text=text.replace("rs.","").replace("rs ","")
    text=re.sub(r"\blakhs?\b","lakh",text)
    text=re.sub(r"\bcrores?\b","crore",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def extract_numbers(text):
    text=str(text or "").replace(",","")
    return [float(x) for x in re.findall(r"(?<!\d)(?:\d+\.\d+|\d+)(?!\d)",text)]

def numbers_match(expected,answer,tolerance=0.01):
    expected_numbers=extract_numbers(expected)
    answer_numbers=extract_numbers(answer)
    if not expected_numbers:
        return None
    if not answer_numbers:
        return False
    used=set()
    for expected_number in expected_numbers:
        matched=False
        for index,answer_number in enumerate(answer_numbers):
            if index in used:
                continue
            if abs(expected_number-answer_number)<=tolerance:
                used.add(index)
                matched=True
                break
        if not matched:
            return False
    return True

def text_match(expected,answer):
    expected_norm=normalize_text(expected)
    answer_norm=normalize_text(answer)
    if not expected_norm:
        return None
    if expected_norm in answer_norm:
        return True
    expected_parts=re.split(r"[;|]",expected_norm)
    expected_parts=[part.strip() for part in expected_parts if part.strip()]
    if expected_parts and all(part in answer_norm for part in expected_parts):
        return True
    return False

def evaluate_expected_answer(expected,answer):
    if not expected:
        return None
    numeric_result=numbers_match(expected,answer)
    text_result=text_match(expected,answer)
    if numeric_result is True:
        if text_result is True or len(extract_numbers(expected))>=1:
            return True
    if text_result is True:
        return True
    return False

def page_match(results,expected_pages):
    if not expected_pages:
        return None
    retrieved={str(get_page(result)) for result in results}
    return any(str(page) in retrieved for page in expected_pages)

def evaluate():
    questions=load_questions()

    print("="*80)
    print("ANSWER EVALUATION")
    print("="*80)
    print(f"Total evaluation questions: {len(questions)}")
    print(f"Retrieval cutoff: Top-{RETRIEVAL_TOP_K}")
    print(f"Context cutoff: Top-{CONTEXT_TOP_K}")
    print("="*80)

    grounded_count=0
    page_success_count=0
    answer_match_count=0
    answer_evaluated_count=0
    failures=[]

    for index,item in enumerate(questions,1):
        question=get_question(item)
        expected_pages=get_expected_pages(item)
        expected_answer=get_expected_answer(item)

        print(f"\n{'-'*80}")
        print(f"Question {index}/{len(questions)}")
        print(f"Q: {question}")
        print(f"Expected page(s): {expected_pages}")
        if expected_answer:
            print(f"Expected answer: {expected_answer}")

        try:
            candidates=hybrid_retrieve(question,top_k=RETRIEVAL_TOP_K)
            context,results=build_context(candidates,candidates,question,top_k=CONTEXT_TOP_K)
            answer=generate_answer(question,(context,results))

            grounded=bool(validate_answer_grounding(answer,results)) if results else False
            expected_page_found=page_match(results,expected_pages)
            answer_match=evaluate_expected_answer(expected_answer,answer)

            if grounded:
                grounded_count+=1
            if expected_page_found:
                page_success_count+=1
            if expected_answer:
                answer_evaluated_count+=1
                if answer_match:
                    answer_match_count+=1

            pages=[get_page(result) for result in results]

            print(f"Answer: {answer}")
            print(f"Grounded: {'PASS' if grounded else 'FAIL'}")
            print(f"Retrieved pages: {pages}")
            print(f"Expected page found: {'PASS' if expected_page_found else 'FAIL'}")

            if expected_answer:
                print(f"Expected answer match: {'PASS' if answer_match else 'FAIL'}")

            if not grounded or not expected_page_found or (expected_answer and not answer_match):
                failures.append({
                    "question":question,
                    "answer":answer,
                    "grounded":grounded,
                    "retrieved_pages":pages,
                    "expected_pages":expected_pages,
                    "expected_answer":expected_answer,
                    "answer_match":answer_match
                })

        except Exception as exc:
            print(f"ERROR: {exc}")
            failures.append({
                "question":question,
                "error":str(exc)
            })

    total=len(questions)
    grounding_rate=grounded_count/total*100 if total else 0
    page_rate=page_success_count/total*100 if total else 0
    answer_rate=answer_match_count/answer_evaluated_count*100 if answer_evaluated_count else 0

    print("\n"+"="*80)
    print("ANSWER EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions       : {total}")
    print(f"Grounded Answers      : {grounded_count}/{total}")
    print(f"Grounding Rate        : {grounding_rate:.2f}%")
    print(f"Expected Page Success : {page_success_count}/{total}")
    print(f"Page Success Rate     : {page_rate:.2f}%")
    if answer_evaluated_count:
        print(f"Expected Answer Match : {answer_match_count}/{answer_evaluated_count}")
        print(f"Answer Match Rate     : {answer_rate:.2f}%")

    passed=page_rate==100.0 and answer_rate==100.0
    print(f"STATUS: {'PASS' if passed else 'FAIL'}")

    if failures:
        print("\nFAILED CASES")
        print("="*80)
        for index,failure in enumerate(failures,1):
            print(f"\n{index}. {failure.get('question','Unknown question')}")
            if "error" in failure:
                print(f"Error: {failure['error']}")
            else:
                print(f"Answer: {failure['answer']}")
                print(f"Grounded: {failure['grounded']}")
                print(f"Retrieved pages: {failure['retrieved_pages']}")
                print(f"Expected pages: {failure['expected_pages']}")
                if failure["expected_answer"]:
                    print(f"Expected answer: {failure['expected_answer']}")
                    print(f"Answer match: {failure['answer_match']}")

    print("="*80)

if __name__=="__main__":
    evaluate()