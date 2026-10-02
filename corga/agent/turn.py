"""What every node works with: the stored state and the dependencies, with named lookups and shared helpers.

Also the deterministic glue between a parsed request and the stored state (window, preferences, unsupported topics).
"""

import re

from typing import Any, NamedTuple
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from collections.abc import Iterable

from pydantic import BaseModel
from langchain_core.messages import AnyMessage, HumanMessage

from corga.agent.deps import Deps
from corga.agent.state import Result, AgentState, ParsedRequest
from corga.data.models import POI
from corga.data.timeutil import at, now
from corga.planning.models import Itinerary, TimeWindow, Preferences, ValidationResult

UNSUPPORTED = {
    "tickets": r"\b(tickets?|admission|entry fees?|prices?|costs?)\b",
    "bookings": r"\b(book|booking|reserve|reservations?)\b",
    "accessibility": r"\b(wheelchair|accessible|accessibility)\b",
    "events": r"\b(events?|concerts?|festivals?)\b",
}


def resolve_window(prev: TimeWindow | None, req: ParsedRequest) -> TimeWindow | None:
    """The time window after this message: the previous one with any stated date or times replaced."""
    if not (req.date or req.start_time or req.end_time):
        return prev  # nothing stated, keep the window
    if prev is None and not (req.date and req.start_time and req.end_time):
        return None  # a first window needs all three parts
    day = req.date or prev.start.date()
    start = at(day, req.start_time or prev.start.time())
    end = at(day, req.end_time or prev.end.time())
    return TimeWindow(start=start, end=end) if end > start else prev  # an inverted edit is ignored


def merge_preferences(prefs: Preferences, req: ParsedRequest) -> Preferences:
    """Preferences stated in this message override the stored ones; unstated ones are kept."""
    return prefs.model_copy(update=req.preference_updates)


def unsupported_topics(text: str) -> list[str]:
    """Topics in the text that the dataset has no facts about: tickets, bookings, accessibility, events."""
    return [topic for topic, pattern in UNSUPPORTED.items() if re.search(pattern, text, re.IGNORECASE)]


def dump(model: BaseModel, **kwargs: Any) -> dict[str, Any]:
    """A model as JSON-ready data, for the result."""
    return model.model_dump(mode="json", **kwargs)


def unique(pois: Iterable[POI]) -> list[POI]:
    """The places in order, each once."""
    return list({poi.id: poi for poi in pois}.values())


def need_info(missing: str) -> AgentState:
    return {"result": {"kind": "need_info", "missing": missing}}


def refusal(reason: str, **details: Any) -> AgentState:
    return {"result": {"kind": "refusal", "reason": reason, **details}}


class Hit(NamedTuple):
    """A place the search returned, with its cosine similarity to the query (higher is closer)."""

    poi: POI
    score: float


class Lookup(NamedTuple):
    """Place names the user gave, split into dataset places and names the dataset does not have."""

    places: list[POI]  # in the order named, each once
    unknown: list[str]


@dataclass(frozen=True)
class Turn:
    """One node call: the state as stored before the node runs, and the dependencies passed to the graph."""

    state: AgentState
    deps: Deps

    # --- the state, by name ---

    @property
    def messages(self) -> list[AnyMessage]:
        return self.state.get("messages", [])

    @property
    def request(self) -> ParsedRequest:
        return self.state["request"]  # set by `parse` before any handler runs

    @property
    def window(self) -> TimeWindow | None:
        return self.state.get("window")

    @property
    def day(self) -> date | None:
        return self.window.start.date() if self.window else None

    @property
    def prefs(self) -> Preferences:
        return self.state.get("preferences") or Preferences()

    @property
    def itinerary(self) -> Itinerary | None:
        return self.state.get("itinerary")

    @property
    def result(self) -> Result:
        return self.state["result"]  # set by the handler before `respond` runs

    @property
    def summary(self) -> str:
        """The session state as short text for the parser prompt: time window, preferences, current stops."""
        window, itinerary = self.window, self.itinerary
        lines = [f"time window: {window.start:%A %d %B %Y %H:%M}-{window.end:%H:%M}" if window else "time window: none"]
        lines.append(f"preferences: {self.prefs.model_dump(exclude_defaults=True) or 'none'}")
        if itinerary and itinerary.stops:
            stops = "; ".join(f"{n}. {s.name}" for n, s in enumerate(itinerary.stops, 1))
            lines.append(f"current itinerary: {stops}")
        else:
            lines.append("current itinerary: none")
        return "\n".join(lines)

    @property
    def question(self) -> str:
        """The question as parsed, else the user's latest message as written."""
        latest = next((m.text for m in reversed(self.messages) if isinstance(m, HumanMessage)), "")
        return self.request.question or latest

    # --- helpers shared by the handlers ---

    def candidates(self, prefs: Preferences | None = None) -> list[POI]:
        """Dataset places passing the preferences (this turn's by default), ranked by the interests."""
        steps = self.deps.steps
        return steps.places(steps.select_candidates.invoke(prefs or self.prefs))

    def search(self, query: str) -> list[Hit]:
        """The dataset places closest to the query, best first, with their similarity scores."""
        return [Hit(self.deps.store.get(pid), score) for pid, score in self.deps.steps.retrieve.invoke(query)]

    def lookup(self, names: list[str]) -> Lookup:
        """Match place names against the dataset."""
        found = {name: self.deps.store.find(name) for name in names}
        places = unique(poi for poi in found.values() if poi is not None)
        return Lookup(places, [name for name, poi in found.items() if poi is None])

    def planned(self, itinerary: Itinerary, result: Result) -> AgentState:
        """Validate, attach to the result, and keep the itinerary in state only if it is valid."""
        validation: ValidationResult = self.deps.steps.validate.invoke(itinerary)
        result |= {"itinerary": dump(itinerary), "validation": dump(validation), "citations": itinerary.poi_ids}
        return {"itinerary": itinerary if validation.ok else self.itinerary, "result": result}

    def weather_range(self) -> tuple[datetime, datetime]:
        """The hours to forecast: a stated horizon, else the plan's window, else the stated day, else the next 24 h."""
        request = self.request
        this_hour = now().replace(minute=0, second=0, microsecond=0)
        if request.horizon_hours:
            return this_hour, this_hour + timedelta(hours=request.horizon_hours)
        windows: tuple[TimeWindow | None, ...] = (self.itinerary.window if self.itinerary else None, self.window)
        for window in windows:
            if window is not None and (request.date is None or request.date == window.start.date()):
                return window.start, window.end
        if request.date is not None:
            return at(request.date, request.start_time or "00:00"), at(request.date, request.end_time or "23:59")
        return this_hour, this_hour + timedelta(hours=24)
