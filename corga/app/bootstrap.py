import re

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from langchain_core.language_models import BaseChatModel

from corga.agent.runtime import Deps
from corga.domain.weather import WeatherProvider
from corga.services.ports import PlaceSearch, PlaceCatalog
from corga.agent.ask_agent import build_agent
from corga.services.places import PlaceFinder
from corga.services.weather import Weather
from corga.services.itineraries import Itineraries
from corga.domain.planning.rules import PlanningRules


def sanitize(value: str) -> str:
    """Make a string safe as one folder name: runs of anything but letters, digits, `.`, `_` and `-` become `-`."""
    return re.sub(r"[^\w.-]+", "-", value).strip("-")


def register_resolvers() -> None:
    """Register CorGA's config interpolations, e.g. `${sanitize:...}`, before Hydra composes the config."""
    if not OmegaConf.has_resolver("sanitize"):
        OmegaConf.register_new_resolver("sanitize", sanitize)


def build_deps(cfg: DictConfig) -> Deps:
    """Build everything the graph needs, once per process, from the config.

    Args:
        cfg: The config Hydra composed.

    Returns:
        The dependencies.
    """
    store: PlaceCatalog = instantiate(cfg.store)
    retriever: PlaceSearch = instantiate(cfg.retrieval, pois=store.all())  # the POIs exist only once the store is built
    rules: PlanningRules = instantiate(cfg.rules)
    llm: BaseChatModel = instantiate(cfg.provider)
    weather: WeatherProvider = instantiate(cfg.weather)

    places = PlaceFinder(store, retriever)

    return Deps(
        llm=llm,
        agent=build_agent(llm),
        places=places,
        itineraries=Itineraries(places, rules),
        weather=Weather(weather),
        structured_method=cfg.structured_method,
    )
