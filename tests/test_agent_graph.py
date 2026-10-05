from agents.router import route_query

def test_calculation_routes_to_calculator():
    assert route_query({"query":"calculate 12 + 8","intent":""})=="calculator"

def test_document_question_routes_to_retrieval():
    assert route_query({"query":"What does the document say about safety?","intent":""})=="retrieval"
