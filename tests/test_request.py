from datetime import time

from tests.conftest import TUE, window
from corga.agent.state import ParsedRequest
from corga.data.timeutil import at
from corga.planning.models import Preferences
from corga.agent.nodes.request import resolve_window, merge_preferences, unsupported_topics


def test_full_window_from_request():
    req = ParsedRequest(intent="plan", date=TUE, start_time=time(11), end_time=time(15))
    assert resolve_window(None, req) == window(TUE, "11:00", "15:00")


def test_partial_request_updates_previous_window():
    prev = window(TUE, "11:00", "15:00")
    assert resolve_window(prev, ParsedRequest(intent="plan", end_time=time(14))).end == at(TUE, "14:00")
    assert resolve_window(prev, ParsedRequest(intent="edit_plan")) == prev


def test_incomplete_first_window_is_none_and_inverted_keeps_previous():
    assert resolve_window(None, ParsedRequest(intent="plan", date=TUE)) is None
    prev = window(TUE, "11:00", "15:00")
    assert resolve_window(prev, ParsedRequest(intent="plan", start_time=time(16))) == prev


def test_preferences_are_sticky():
    prefs = Preferences(interests=["history"])
    merged = merge_preferences(prefs, ParsedRequest(intent="edit_plan", stop_index=2, with_child=True))
    assert merged.interests == ["history"] and merged.with_child is True
    assert merge_preferences(merged, ParsedRequest(intent="plan", interests=["Art"])).interests == ["art"]


def test_unsupported_topics():
    assert unsupported_topics("Can I buy tickets online?") == ["tickets"]
    assert unsupported_topics("Is there a bookshop near the Liston?") == []
    assert unsupported_topics("Can I reserve a table?") == ["bookings"]


def test_times_with_an_offset_become_local_and_checkpoint():
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    # Qwen through vLLM writes offsets; pydantic would keep them as its own TzInfo, which msgpack cannot encode.
    req = ParsedRequest.model_validate({"intent": "plan", "start_time": "13:00:00+03:00", "end_time": "17:00"})
    assert (req.start_time, req.end_time) == (time(13), time(17))
    serde = JsonPlusSerializer()
    assert serde.loads_typed(serde.dumps_typed(req)) == req


def test_a_stated_setting_replaces_the_stored_one():
    outdoor = merge_preferences(Preferences(), ParsedRequest(intent="plan", setting="outdoor"))
    assert outdoor.setting == "outdoor" and outdoor.indoor().setting == "indoor"
    assert merge_preferences(outdoor, ParsedRequest(intent="edit_plan", stop_index=2)).setting == "outdoor"  # kept


def test_preference_updates_hold_only_what_the_message_states():
    assert ParsedRequest(intent="edit_plan", stop_index=2).preference_updates == {}
    stated = ParsedRequest(intent="edit_plan", setting="outdoor", with_child=False, interests=[" Food "])
    assert stated.preference_updates == {"setting": "outdoor", "with_child": False, "interests": [" Food "]}
    assert ParsedRequest(intent="plan", interests=["", " "]).preference_updates == {}  # blank interests state nothing


def test_interests_are_normalized_by_the_preferences():
    merged = merge_preferences(Preferences(), ParsedRequest(intent="plan", interests=[" Art ", "", "HISTORY"]))
    assert merged.interests == ["art", "history"]
    assert Preferences(interests=["Museums "]).interests == ["museums"]
