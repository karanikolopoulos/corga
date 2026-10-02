import json

import pytest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from langchain_core.embeddings import DeterministicFakeEmbedding

from corga import CONF_DIR, DATASET_PATH
from corga.app import sanitize, register_resolvers
from corga.domain.planning.rules import PlanningRules


def omegaconf_values(value):
    """Every DictConfig/ListConfig inside plain dicts and lists, at any depth."""
    if isinstance(value, (DictConfig, ListConfig)):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from omegaconf_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from omegaconf_values(item)


register_resolvers()


def load(name, overrides=()):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def test_chat_defaults():
    cfg = load("chat")
    assert cfg.retrieval.qdrant_url is None
    assert cfg.weather._target_ == "corga.adapters.weather.open_meteo.OpenMeteoWeather"
    assert "prompts" not in cfg  # the templates are files in corga/agent/templates/, not settings
    assert cfg.structured_method == "function_calling"


@pytest.mark.parametrize("provider", ["vllm", "openai", "anthropic"])
def test_run_folder_is_one_level_per_part(provider):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="chat", overrides=[f"provider={provider}"], return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)  # what @hydra.main does; lets ${hydra:...} resolve
    run_dir = OmegaConf.to_container(cfg.hydra.run, resolve=True)["dir"]
    assert len(run_dir.split("/")) == 5  # runs/chat/<provider>/<model>/<timestamp>, whatever the model id holds


def test_sanitize_makes_one_folder_name():
    assert sanitize("org/model-1.5 instruct") == "org-model-1.5-instruct"


def test_debug_mode_writes_no_run_folder():
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        normal = compose(config_name="chat", return_hydra_config=True).hydra
        debug = compose(config_name="chat", overrides=["hydra=debug"], return_hydra_config=True).hydra
    assert normal.output_subdir == ".hydra" and list(normal.job_logging.handlers) == ["file"]
    assert (debug.run.dir, debug.output_subdir) == (".", None)
    assert "handlers" not in debug.job_logging and "handlers" not in debug.hydra_logging.root  # Hydra's `disabled`


def test_overrides_switch_provider_and_weather():
    cfg = load("chat", ["provider=anthropic", "weather=fixture"])
    assert cfg.provider._target_ == "langchain_anthropic.ChatAnthropic"
    assert instantiate(cfg.weather).source == "fixture"


def test_vllm_options_reach_the_request_body():
    llm = instantiate(load("chat").provider)  # no request is made
    assert not llm.model_kwargs  # nothing that goes to the OpenAI SDK as an unknown keyword
    # Encoded the way the SDK encodes the request body; a nested DictConfig compares equal to a dict but fails here.
    assert json.loads(json.dumps(llm.extra_body)) == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("provider", ["vllm", "openai", "anthropic"])
def test_clients_get_no_omegaconf_objects(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = instantiate(load("chat", [f"provider={provider}"]).provider)
    assert not list(omegaconf_values(vars(llm)))


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_secrets_are_interpolations_not_values(provider):
    cfg = load("chat", [f"provider={provider}"])
    key = OmegaConf.to_container(cfg.provider, resolve=False)["api_key"]
    assert key == f"${{oc.env:{provider.upper()}_API_KEY}}"


def test_rules_default_to_the_code_defaults_and_can_be_overridden():
    assert instantiate(load("chat").rules) == PlanningRules()  # the YAML values match the dataclass defaults
    rules = instantiate(load("chat", ["+max_stops=4", "+speed_kmh=5.5"]).rules)  # top-level, like +model=...
    assert (rules.max_stops, rules.walking.speed_kmh) == (4, 5.5)


def test_the_store_is_built_from_config():
    assert len(instantiate(load("chat").store).all()) == 50  # the bundled dataset by default
    assert len(instantiate(load("chat", [f"+store.path={DATASET_PATH}"]).store).all()) == 50


def test_the_retriever_is_built_from_config(store):
    fake = DeterministicFakeEmbedding(size=32)  # replaces the embedding group's model, which is not loaded
    retriever = instantiate(load("chat", ["retrieval.k=6"]).retrieval, pois=store.all(), embeddings=fake)
    assert (retriever.size, retriever.k, retriever.store.embeddings) == (50, 6, fake)
