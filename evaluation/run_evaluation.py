import os
import subprocess
import sys
from datetime import datetime

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALUATION_DIR=os.path.join(ROOT,"evaluation")
EVALUATORS=[
    ("Retrieval Evaluation","evaluate_retrieval.py"),
    ("Reranker Evaluation","evaluate_reranker.py"),
    ("Answer Evaluation","evaluate_answer.py"),
]

def run_evaluator(name,filename):
    path=os.path.join(EVALUATION_DIR,filename)
    print("\n"+"="*80)
    print(name.upper())
    print("="*80)
    try:
        process=subprocess.run(
            [sys.executable,path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output=process.stdout
        if process.stderr:
            output+=process.stderr
        print(output)
        passed="STATUS: PASS" in output
        return {
            "name":name,
            "file":filename,
            "passed":passed,
            "returncode":process.returncode,
            "output":output,
        }
    except Exception as exc:
        print(f"ERROR: {exc}")
        return {
            "name":name,
            "file":filename,
            "passed":False,
            "returncode":-1,
            "output":str(exc),
        }

def main():
    started=datetime.now()
    print("="*80)
    print("DOCUMENT INTELLIGENCE PLATFORM")
    print("UNIFIED EVALUATION RUNNER")
    print("="*80)
    print(f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
    results=[run_evaluator(name,filename) for name,filename in EVALUATORS]
    print("\n"+"="*80)
    print("UNIFIED EVALUATION SUMMARY")
    print("="*80)
    for result in results:
        status="PASS" if result["passed"] else "FAIL"
        print(f"{result['name']:<30}: {status}")
    overall_pass=all(result["passed"] for result in results)
    finished=datetime.now()
    print("-"*80)
    print(f"Started : {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {finished.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {finished-started}")
    print("="*80)
    print(f"OVERALL STATUS: {'PASS' if overall_pass else 'FAIL'}")
    print("="*80)
    return 0 if overall_pass else 1

if __name__=="__main__":
    sys.exit(main())