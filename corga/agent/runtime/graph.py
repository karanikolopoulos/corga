from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from corga.agent.nodes import ask, build, route, choose, refuse, extract, resolve, respond, forecast, after_resolve
from corga.agent.state import TurnState
from corga.agent.nodes.errors import rephrase, no_answer, forecast_unavailable
from corga.agent.runtime.deps import Deps
from corga.domain.planning.models import Itinerary, TimeWindow, Preferences

CHECKPOINTED = (TimeWindow, Preferences, Itinerary)  # our types the checkpointer may restore: the sticky state


def build_graph() -> CompiledStateGraph:
    """Build the graph: route, then plan, ask or refuse, then respond.

    The plan path runs extract, resolve, forecast (when asked) and build. A node that cannot read the message, an agent
    that does not answer and a forecast that fails each hand over to `respond` through an error handler. The
    dependencies come with every call: `graph.invoke(inputs, config, context=deps)`.

    Returns:
        The graph, its state kept per `thread_id` in an in-memory checkpointer.
    """
    graph = StateGraph(TurnState, context_schema=Deps)
    graph.add_node(route, error_handler=rephrase)
    graph.add_node(extract, error_handler=rephrase)
    graph.add_node(resolve)
    graph.add_node(forecast, error_handler=forecast_unavailable)
    graph.add_node(build)
    graph.add_node(ask, error_handler=no_answer)
    graph.add_node(refuse)
    graph.add_node(respond)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", choose)  # targets come from `choose`'s Literal return type
    graph.add_edge("extract", "resolve")
    graph.add_conditional_edges("resolve", after_resolve)
    graph.add_edge("forecast", "build")
    graph.add_edge("build", "respond")
    graph.add_edge("ask", "respond")
    graph.add_edge("refuse", END)
    graph.add_edge("respond", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED)

    return graph.compile(checkpointer=InMemorySaver(serde=serde))
