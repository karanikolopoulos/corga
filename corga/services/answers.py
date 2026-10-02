"""What the dataset says about places: answers to questions, and recommendations for the preferences."""

import re

from datetime import date

from corga.services.steps import PlanningSteps
from corga.services.places import PlaceFinder, unique
from corga.services.results import Fact, Item, AnswerResult, RecommendationResult
from corga.domain.planning.models import Preferences

UNSUPPORTED = {
    "tickets": r"\b(tickets?|admission|entry fees?|prices?|costs?)\b",
    "bookings": r"\b(book|booking|reserve|reservations?)\b",
    "accessibility": r"\b(wheelchair|accessible|accessibility)\b",
    "events": r"\b(events?|concerts?|festivals?)\b",
}


def unsupported_topics(text: str) -> list[str]:
    """Topics in the text that the dataset has no facts about: tickets, bookings, accessibility, events."""
    return [topic for topic, pattern in UNSUPPORTED.items() if re.search(pattern, text, re.IGNORECASE)]


class Answers:
    """The dataset-facts use cases, over the place finder and the planning steps; held by `Deps`."""

    def __init__(self, places: PlaceFinder, steps: PlanningSteps):
        self.places = places
        self.steps = steps

    def answer_question(self, question: str, named: list[str]) -> AnswerResult:
        """Facts about the places the user named plus the closest matches to the question, all from the dataset."""
        lookup = self.places.lookup(named)
        # TODO(calibrate): drop hits below a minimum similarity so off-topic questions cite nothing, instead of the
        # 4 nearest POIs; pick the threshold from MiniLM scores on on-topic vs off-topic questions (e.g. 'aquarium').
        hits = self.places.search(question)
        scores = {hit.poi.id: hit.score for hit in hits}
        pois = unique([*lookup.places, *(hit.poi for hit in hits)])  # named first
        return AnswerResult(
            question=question,
            facts=[Fact.of(poi, scores.get(poi.id)) for poi in pois],
            unsupported_topics=unsupported_topics(question),
            unknown_places=lookup.unknown,
        )

    def recommend_places(self, prefs: Preferences, day: date | None) -> RecommendationResult:
        """The top candidates for the preferences, open on `day` when there is one."""
        pois = self.steps.recommendations(prefs, day)
        return RecommendationResult(preferences=prefs, items=[Item.of(poi, prefs) for poi in pois])
