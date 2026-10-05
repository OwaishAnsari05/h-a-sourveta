from pathlib import Path
from agents.router import route_query
from agents.graph import agent_graph

def test_calculation_route():
    assert route_query({"query":"What was the change in total income between FY 2023-24 and FY 2024-25?","intent":"general"})=="calculator"

def test_general_route():
    assert route_query({"query":"What does the document say about laminar flow equipment?","intent":"general"})=="retrieval"

def test_retrieval_node_is_evidence_only():
    text=Path("agents/nodes/retrieval.py").read_text(encoding="utf-8")
    assert "ask_question" not in text
    assert "generate_answer" not in text
    assert "retrieve_evidence" in text

def test_graph_has_single_generation_after_retrieval():
    text=Path("agents/graph.py").read_text(encoding="utf-8")
    assert 'graph.add_edge("router","retrieval")' in text
    assert '"retrieval":"generation"' in text
    assert '"calculator":"calculator"' in text