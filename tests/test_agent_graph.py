from agents.graph import agent_graph

query="What happened to the companys total income between FY 2023-24 and FY 2024-25?"
result=agent_graph.invoke({"query":query,"document_id":"tata_annual_report_2024_25"})
print("INTENT:",result.get("intent"))
print("ROUTE:",result.get("route"))
print("ANSWER:",result.get("answer"))
print("VALIDATED:",result.get("validated"))
print("ERRORS:",result.get("validation_errors"))
print("CITATIONS:",result.get("citations"))