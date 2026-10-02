"""Composition root: builds the graph's dependencies from the Hydra config."""

import re

from typing import Any

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from corga import DATASET_PATH
from corga.agent.deps import Deps
from corga.data.store import POIStore
from corga.agent.templates import parse_prompt, respond_prompt
from corga.retrieval.index import build_index
from corga.retrieval.search import Retriever


def sanitize(value: str) -> str:
    """A string safe as one folder name: runs of anything but letters, digits, `.`, `_` and `-` become `-`."""
    return re.sub(r"[^\w.-]+", "-", value).strip("-")


def register_resolvers() -> None:
    """Config interpolations CorGA adds, e.g. `${sanitize:${provider.model}}`; call before Hydra composes."""
    if not OmegaConf.has_resolver("sanitize"):
        OmegaConf.register_new_resolver("sanitize", sanitize)


def build(node: DictConfig) -> Any:
    """Instantiate a `_target_` node with plain dicts and lists for nested options."""
    return instantiate(node, _convert_="all")


def build_deps(cfg: DictConfig) -> Deps:
    """Everything the graph needs, built once per process from the composed config."""
    store = POIStore.from_json(DATASET_PATH)
    embeddings = build(cfg.embedding)
    index = build_index(store.all(), embeddings, url=cfg.retrieval.qdrant_url, api_key=cfg.retrieval.qdrant_api_key)
    return Deps(
        llm=build(cfg.provider),
        store=store,
        retriever=Retriever(index, size=len(store.all()), k=cfg.retrieval.k),
        weather=build(cfg.weather),
        rules=build(cfg.rules),
        structured_method=cfg.structured_method,
        parse_prompt=parse_prompt(cfg.prompts.parse),
        respond_prompt=respond_prompt(cfg.prompts.respond),
    )
