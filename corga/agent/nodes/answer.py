"""Dataset-facts nodes: answers about places, and recommendations for the preferences."""

from langgraph.runtime import Runtime

from corga.agent.guide import unique
from corga.agent.state import Update, AgentState
from corga.agent.results import Fact, Item, AnswerResult, RecommendationResult
from corga.agent.runtime.deps import Deps
from corga.agent.nodes.request import unsupported_topics


def answer(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """Facts about the places the user named plus the closest matches to the question, all from the dataset."""
    guide = runtime.context.guide
    question = state.question
    named = guide.lookup(state.request.requested_places)
    # TODO(calibrate): drop hits below a minimum similarity so off-topic questions cite nothing, instead of the
    # 4 nearest POIs; pick the threshold from MiniLM scores on on-topic vs off-topic questions (e.g. 'aquarium').
    hits = guide.search(question)
    scores = {hit.poi.id: hit.score for hit in hits}
    pois = unique([*named.places, *(hit.poi for hit in hits)])  # named first
    result = AnswerResult(
        question=question,
        facts=[Fact.of(p, scores.get(p.id)) for p in pois],
        unsupported_topics=unsupported_topics(question),
        unknown_places=named.unknown,
    )
    return {"result": result}


def recommend(state: AgentState, runtime: Runtime[Deps]) -> Update:
    """The top candidates for the preferences, open on the planned day when there is one."""
    pois = runtime.context.guide.recommendations(state.preferences, state.day)
    items = [Item.of(p, state.preferences) for p in pois]
    return {"result": RecommendationResult(preferences=state.preferences, items=items)}
