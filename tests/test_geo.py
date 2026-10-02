from corga.planning.geo import WalkingModel, haversine_km
from corga.planning.rules import PlanningRules


def test_haversine_old_fortress_to_achilleion(store):
    km = haversine_km(store.get("CF-001").location, store.get("CF-021").location)
    assert 7.0 < km < 7.1


def test_walking_minutes_are_rounded_up(store):
    walking = WalkingModel()
    assert walking.minutes(store.get("CF-001").location, store.get("CF-021").location) == 122
    assert walking.minutes(store.get("CF-001").location, store.get("CF-002").location) == 18
    assert walking.minutes(store.get("CF-001").location, store.get("CF-001").location) == 0


def test_rules_state_their_assumptions():
    text = " ".join(PlanningRules().assumptions())
    assert "1.3" in text and "4.5 km/h" in text and "fixtures" in text
