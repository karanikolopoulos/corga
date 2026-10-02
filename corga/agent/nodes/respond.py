"""The last node of every turn, and the refusal handler whose result it phrases."""

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from corga.agent.state import Update, AgentState
from corga.services.results import RefusalResult
from corga.agent.runtime.deps import Deps


def refuse(state: AgentState) -> Update:
    reason = "fabrication_or_ignore_dataset" if state.parsed.wants_fabrication else "out_of_scope"
    return {"result": RefusalResult(reason=reason)}


def respond(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """A few framing sentences from the LLM, followed by the deterministic rendering of the result."""
    deps = runtime.context
    system = deps.prompts["respond"].render()
    framing: BaseMessage = deps.llm.invoke(
        [SystemMessage(system), HumanMessage(f"RESULT:\n{state.result.model_dump_json()}")]
    )
    body = deps.prompts["reply"].render(result=state.result).strip()  # times, legs, checks, forecasts, citations
    return {"messages": [AIMessage(content=f"{framing.text}\n\n{body}".strip())]}
