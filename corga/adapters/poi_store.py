import re
import json
import difflib

from typing import Self
from pathlib import Path

from corga import DATASET_PATH
from corga.domain.places import POI


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class POIStore:
    def __init__(self, pois: list[POI]):
        self._by_id = {p.id: p for p in pois}
        self._by_name = {_norm(p.name): p for p in pois}

    @classmethod
    def from_json(cls, path: str | Path = DATASET_PATH) -> Self:
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        return cls([POI.model_validate(p) for p in data["pois"]])

    def get(self, poi_id: str) -> POI:
        return self._by_id[poi_id]

    def all(self) -> list[POI]:
        return list(self._by_id.values())

    def find(self, name: str) -> POI | None:
        """Find a place by name, allowing small typos.

        Args:
            name: The name as the user wrote it.

        Returns:
            The place, or None if no name is close enough.
        """
        key = _norm(name)
        if key in self._by_name:
            return self._by_name[key]

        match = difflib.get_close_matches(key, list(self._by_name), n=1, cutoff=0.85)

        return self._by_name[match[0]] if match else None
