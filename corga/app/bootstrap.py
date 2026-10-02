"""Composition root: builds the graph's dependencies from the Hydra config."""

import re

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel

from corga.agent.runtime import Deps
from corga.domain.weather import WeatherProvider
from corga.services.ports import PlaceSearch, PlaceCatalog
from corga.services.steps import PlanningSteps
from corga.services.places import PlaceFinder
from corga.services.answers import Answers
from corga.services.itineraries import Itineraries
from corga.domain.planning.rules import PlanningRules


def sanitize(value: str) -> str:
    """A string safe as one folder name: runs of anything but letters, digits, `.`, `_` and `-` become `-`."""
    return re.sub(r"[^\w.-]+", "-", value).strip("-")


def register_resolvers() -> None:
    """Config interpolations CorGA adds, e.g. `${sanitize:${provider.model}}`; call before Hydra composes."""
    if not OmegaConf.has_resolver("sanitize"):
        OmegaConf.register_new_resolver("sanitize", sanitize)


def build_deps(cfg: DictConfig) -> Deps:
    """Everything the graph needs, built once per process from the composed config."""
    store: PlaceCatalog = instantiate(cfg.store)
    retriever: PlaceSearch = instantiate(cfg.retrieval, pois=store.all())  # the POIs exist only once the store is built
    rules: PlanningRules = instantiate(cfg.rules)
    llm: BaseChatModel = instantiate(cfg.provider)
    weather: WeatherProvider = instantiate(cfg.weather)
    places = PlaceFinder(store, retriever)
    steps = PlanningSteps(store, rules, ranker=retriever.rank)
    return Deps(
        llm=llm,
        answers=Answers(places, steps),
        itineraries=Itineraries(steps, places),
        weather=weather,
        structured_method=cfg.structured_method,
    )
