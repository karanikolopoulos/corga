import logging

from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage

from corga.agent.state import Update, TurnState
from corga.services.results import RefusalResult
from corga.agent.runtime.deps import Deps

log = logging.getLogger(__name__)


def refuse(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Refuse with a fixed reply and no LLM call.

    The turn's message leaves the history and the plan stays.

    Args:
        state: The turn's state.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The message's removal, the refusal and the reply.
    """
    reason = "fabrication_or_ignore_dataset" if state.path == "attack" else "out_of_scope"
    message = next(m for m in reversed(state.messages) if isinstance(m, HumanMessage))
    reply = runtime.context.prompts["refuse"].render(reason=reason).strip()

    log.info("refused: %s", reason)

    return {"messages": [RemoveMessage(id=message.id)], "result": RefusalResult(reason=reason), "reply": reply}


def respond(state: TurnState, runtime: Runtime[Deps]) -> Update:
    """Frame the result in a few sentences from the LLM, then render it.

    The agent's answer gets no framing.

    Args:
        state: The turn's state, with its result.
        runtime: The graph's runtime, the dependencies as its context.

    Returns:
        The reply, as a message and as text.
    """
    deps = runtime.context
    body = deps.prompts["reply"].render(result=state.result).strip()  # times, legs, checks, forecasts, citations

    if state.result.kind == "answer":
        return {"messages": [AIMessage(content=body)], "reply": body}

    system = deps.prompts["respond"].render()
    framing: BaseMessage = deps.llm.invoke(
        [SystemMessage(system), HumanMessage(f"RESULT:\n{state.result.model_dump_json()}")]
    )
    text = f"{framing.text}\n\n{body}".strip()

    return {"messages": [AIMessage(content=text)], "reply": text}
