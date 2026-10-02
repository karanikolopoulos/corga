from tests.conftest import MON
from corga.domain.planning.models import Preferences


def test_search_returns_the_closest_places(places):
    assert [hit.poi.id for hit in places.search("fortress")] == ["CF-001", "CF-002"]


def test_lookup_splits_names_into_dataset_places_and_unknown_names(places):
    lookup = places.lookup(["old fortress", "Corfu Aquarium", "Old Fortress"])

    assert [poi.id for poi in lookup.places] == ["CF-001"]  # each place once, in the order named
    assert lookup.unknown == ["Corfu Aquarium"]


def test_recommendations_for_the_preferences_keep_places_open_that_day(places):
    picked = places.recommend(Preferences(interests=["art", "museums"]), MON)

    assert picked and not {"CF-014", "CF-017", "CF-031"} & {poi.id for poi in picked}
