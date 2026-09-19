from agents.nodes.calculator import calculator_node
from agents.nodes.retrieval import retrieval_node

query="What happened to the companys total income between FY 2023-24 and FY 2024-25?"
state={"query":query,"document_id":"tata_annual_report_2024_25"}
state.update(retrieval_node(state))
result=calculator_node(state)

print("ANSWER:",result.get("answer"))
print("ERROR:",result.get("error"))