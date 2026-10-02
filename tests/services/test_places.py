def test_search_returns_places_best_first_with_their_scores(places):
    hits = places.search("Venetian fortress")
    assert [(hit.poi.name, hit.score) for hit in hits] == [("Old Fortress", 0.74), ("New Fortress", 0.61)]


def test_lookup_splits_names_into_dataset_places_and_unknown_names(places):
    lookup = places.lookup(["old fortress", "Corfu Aquarium", "Old Fortress"])
    assert [poi.id for poi in lookup.places] == ["CF-001"]  # each place once, in the order named
    assert lookup.unknown == ["Corfu Aquarium"]
