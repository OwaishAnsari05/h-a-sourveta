from langgraph.graph import StateGraph,START,END
from agents.state import AgentState
from agents.router import router_node
from agents.nodes.query_resolver import query_resolver_node
from agents.nodes.triage import triage_node
from agents.nodes.retrieval import retrieval_node
from agents.nodes.generation import generation_node
from agents.nodes.calculator import calculator_node
from agents.nodes.validator import validator_node
from agents.nodes.citation import citation_node

def build_graph():
    graph=StateGraph(AgentState)
    graph.add_node("query_resolver",query_resolver_node)
    graph.add_node("triage",triage_node)
    graph.add_node("router",router_node)
    graph.add_node("retrieval",retrieval_node)
    graph.add_node("generation",generation_node)
    graph.add_node("calculator",calculator_node)
    graph.add_node("validator",validator_node)
    graph.add_node("citation",citation_node)
    graph.add_edge(START,"query_resolver")
    graph.add_edge("query_resolver","triage")
    graph.add_edge("triage","router")
    graph.add_conditional_edges("router",lambda state: state.get("route","retrieval"),{"retrieval":"retrieval","calculator":"retrieval"})
    graph.add_conditional_edges("retrieval",lambda state: state.get("route","retrieval"),{"retrieval":"generation","calculator":"calculator"})
    graph.add_edge("generation","validator")
    graph.add_edge("calculator","validator")
    graph.add_edge("validator","citation")
    graph.add_edge("citation",END)
    return graph.compile()

agent_graph=build_graph()