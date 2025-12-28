from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.lead_capture import lead_capture_node
from app.nodes.qa import qa_node
from app.nodes.router import route
from app.state import ChatState


def build_graph() -> StateGraph:
    graph = StateGraph(ChatState)

    graph.add_node("qa", qa_node)
    graph.add_node("lead_capture", lead_capture_node)

    graph.add_conditional_edges(
        START,
        route,
        {
            "qa": "qa",
            "lead_capture": "lead_capture",
        },
    )

    graph.add_edge("qa", END)
    graph.add_edge("lead_capture", END)

    return graph


app = build_graph().compile()
