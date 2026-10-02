"""The dataset-facts use cases on the in-memory dataset store and a plain search fake: no LangChain, no graph."""

import pytest

from tests.conftest import MON
from corga.services.answers import Answers, unsupported_topics
from corga.domain.planning.models import Preferences


@pytest.fixture
def answers(places, steps):
    return Answers(places, steps)


def test_answer_question_puts_the_named_places_first_then_the_closest(answers):
    result = answers.answer_question("Can I buy tickets for the Achilleion?", ["Achilleion Palace", "Corfu Aquarium"])
    assert [fact.id for fact in result.facts] == ["CF-021", "CF-001", "CF-002"]
    assert (result.facts[0].score, result.facts[1].score) == (None, 0.74)  # named, not returned by the search
    assert result.unsupported_topics == ["tickets"] and result.unknown_places == ["Corfu Aquarium"]


def test_unsupported_topics():
    assert unsupported_topics("Can I buy tickets online?") == ["tickets"]
    assert unsupported_topics("Is there a bookshop near the Liston?") == []
    assert unsupported_topics("Can I reserve a table?") == ["bookings"]


def test_recommend_places_keeps_only_places_open_on_the_day(answers):
    result = answers.recommend_places(Preferences(interests=["art", "museums"]), MON)
    assert result.items and not {"CF-014", "CF-017", "CF-031"} & {item.id for item in result.items}
