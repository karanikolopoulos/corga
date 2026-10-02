from tests.conftest import RULES
from corga.services.steps import PlanningSteps
from corga.domain.planning.models import Preferences


def test_candidates_are_ranked_by_the_given_ranker(store):
    steps = PlanningSteps(store, RULES, ranker=lambda query, allowed_ids: sorted(allowed_ids, reverse=True))
    ids = [poi.id for poi in steps.select_candidates(Preferences(interests=["history"]))]
    assert len(ids) > 1 and ids == sorted(ids, reverse=True)
