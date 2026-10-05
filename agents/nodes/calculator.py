import ast,operator,re
from typing import Any
from agents.state import AgentState

_OPERATORS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow}

def _eval(node:ast.AST)->float:
    if isinstance(node,ast.Expression): return _eval(node.body)
    if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)): return float(node.value)
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        value=_eval(node.operand); return value if isinstance(node.op,ast.UAdd) else -value
    if isinstance(node,ast.BinOp) and type(node.op) in _OPERATORS:
        left,right=_eval(node.left),_eval(node.right)
        if isinstance(node.op,ast.Div) and right==0: raise ValueError("Division by zero.")
        return _OPERATORS[type(node.op)](left,right)
    raise ValueError("Unsupported calculation.")

def _expression(query:str)->str|None:
    match=re.search(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?:\s*[+\-*/^]\s*-?\d+(?:\.\d+)?)+(?![A-Za-z0-9_.])",query)
    if not match: return None
    return match.group(0).replace("^","**")

def calculator_node(state:AgentState)->dict[str,Any]:
    expression=_expression(state.get("query",""))
    if not expression: return {"answer":"","error":"No supported arithmetic expression found."}
    try: value=_eval(ast.parse(expression,mode="eval"))
    except (SyntaxError,ValueError,ZeroDivisionError,OverflowError): return {"answer":"","error":"Unable to evaluate the calculation."}
    formatted=f"{value:.10g}"
    return {"answer":formatted,"error":None}
