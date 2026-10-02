import pytest

from tests.conftest import TUE, WED, RULES, window
from corga.services.results import RefusalResult
from corga.services.itineraries import Itineraries
from corga.domain.planning.models import Preferences
from corga.domain.planning.refill import Change

HISTORY = Preferences(interests=["history"])


@pytest.fixture
def itineraries(places):
    return Itineraries(places, RULES)


def test_check_places_refuses_places_not_in_the_dataset(itineraries):
    result = itineraries.check_places(window(WED, "13:00", "16:00"), ["Old Fortress", "Corfu Aquarium"])

    assert result == RefusalResult(reason="not_in_dataset", names=["Corfu Aquarium"])


@pytest.fixture
def history_walk(itineraries):
    """Build the Tuesday 11:00-15:00 history plan."""
    return itineraries.plan(window(TUE, "11:00", "15:00"), HISTORY).itinerary


def test_plan_asks_for_a_stored_plan_and_a_stop_it_has(itineraries, history_walk):
    second, sixth = Change(replace=frozenset({2})), Change(replace=frozenset({6}))

    assert itineraries.plan(history_walk.window, HISTORY, None, second).missing == "an existing itinerary to edit"
    assert itineraries.plan(history_walk.window, HISTORY, history_walk, sixth).missing == "which stop to change (1-5)"
