from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from corga.agent.deps import Deps
from corga.agent.nodes import plan, parse, route, answer, refuse, respond, weather, edit_plan, recommend
from corga.agent.state import AgentState

HANDLERS = (answer, recommend, plan, edit_plan, weather, refuse)

AgentGraph = CompiledStateGraph[AgentState, Deps, AgentState, AgentState]  # state, context, input, output


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> AgentGraph:
    """parse → one handler (by intent) → respond.

    State persists per `thread_id` through the checkpointer; the dependencies are not part of the graph and are
    passed on every call: `graph.invoke(inputs, config, context=deps)`.
    """
    graph = StateGraph(AgentState, context_schema=Deps)
    graph.add_node(parse)
    graph.add_node(respond)
    for handler in HANDLERS:
        graph.add_node(handler)
        graph.add_edge(handler.__name__, "respond")
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route)  # targets come from `route`'s Literal return type
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
