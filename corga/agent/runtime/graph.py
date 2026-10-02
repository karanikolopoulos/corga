from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from corga.agent.nodes import plan, parse, answer, refuse, respond, weather, edit_plan, recommend
from corga.agent.state import Intent, AgentState, ParsedRequest
from corga.agent.runtime.deps import Deps

Handler = Literal["answer", "recommend", "plan", "edit_plan", "weather", "refuse"]
ROUTES: dict[Intent, Handler] = {
    "question": "answer",
    "recommend": "recommend",
    "plan": "plan",
    "edit_plan": "edit_plan",
    "weather": "weather",
    "out_of_scope": "refuse",
}
HANDLERS = (answer, recommend, plan, edit_plan, weather, refuse)

AgentGraph = CompiledStateGraph[AgentState, Deps, AgentState, AgentState]  # state, context, input, output


def route(state: AgentState) -> Handler:
    """The handler for this turn; a request to invent or ignore the dataset is refused whatever its intent."""
    request: ParsedRequest = state.request
    return "refuse" if request.wants_fabrication else ROUTES[request.intent]


def build_graph() -> AgentGraph:
    """parse → one handler (by intent) → respond.

    State persists per `thread_id` in an in-memory checkpointer; the dependencies are not part of the graph and are
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
    return graph.compile(checkpointer=InMemorySaver())
