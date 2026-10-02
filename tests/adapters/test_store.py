def test_aliases_are_mapped(store):
    casa = store.get("CF-007")

    assert casa.family_suitable is True
    assert casa.opening_hours["mon"] == "10:00-17:00"


def test_find_by_name(store):
    assert store.find("Mon Repos Museum of Palaiopolis").id == "CF-022"
    assert store.find("old fortress").id == "CF-001"
    assert store.find("Corfu Aquarium") is None
