from agents.nodes.calculator import calculator_node

def test_addition():
    result=calculator_node({"query":"calculate 12 + 8"})
    assert result["error"] is None
    assert result["answer"]=="20"

def test_expression_precedence():
    result=calculator_node({"query":"compute 10 + 5 * 2"})
    assert result["error"] is None
    assert result["answer"]=="20"

def test_division_by_zero():
    result=calculator_node({"query":"calculate 10 / 0"})
    assert result["answer"]==""
    assert result["error"]
