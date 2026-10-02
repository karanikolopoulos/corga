from datetime import date


def test_loads_all_pois(store):
    assert len(store.all()) == 50


def test_null_hours_stay_none(store):
    assert store.get("CF-004").opening_hours is None  # Saint Spyridon Church


def test_aliases_are_mapped(store):
    casa = store.get("CF-007")
    assert casa.family_suitable is True
    assert casa.opening_hours["mon"] == "10:00-17:00"


def test_new_fortress_closed_on_tuesday(store):
    assert not store.get("CF-002").opening_hours.is_open_on(date(2026, 10, 6))


def test_find_by_name(store):
    assert store.find("Mon Repos Museum of Palaiopolis").id == "CF-022"
    assert store.find("old fortress").id == "CF-001"
    assert store.find("Corfu Aquarium") is None
