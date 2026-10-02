"""Dataset-facts nodes: answers about places, and recommendations for the preferences."""

from langgraph.runtime import Runtime

from corga.agent.state import Update, AgentState
from corga.agent.runtime.deps import Deps


def answer(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Facts about the places the user named plus the closest matches to the question, all from the dataset."""
    result = runtime.context.answers.answer_question(state.question, state.parsed.requested_places)
    return {"result": result}


def recommend(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """The top candidates for the preferences, open on the planned day when there is one."""
    return {"result": runtime.context.answers.recommend_places(state.preferences, state.day)}
