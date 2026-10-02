"""The last node of every turn, and the refusal handler whose result it phrases."""

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, BaseMessage

from corga.agent.state import Update, AgentState
from corga.agent.results import RefusalResult
from corga.agent.templates import render_result
from corga.agent.runtime.deps import Deps


def refuse(state: AgentState) -> Update:
    reason = "fabrication_or_ignore_dataset" if state.request.wants_fabrication else "out_of_scope"
    return {"result": RefusalResult(reason=reason)}


def respond(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """A few framing sentences from the LLM, followed by the deterministic rendering of the result."""
    deps = runtime.context
    payload = state.result.model_dump_json()
    framing: BaseMessage = deps.llm.invoke(deps.respond_prompt.invoke({"result": payload}))
    body = render_result(state.result)
    return {"messages": [AIMessage(content=f"{framing.text}\n\n{body}".strip())]}
