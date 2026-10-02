"""Interactive chat: `python -m corga [provider=openai] [+model=...] [weather=fixture] [hydra=debug] ...`."""

import sys
import signal
import readline  # noqa: F401  (imported for its side effect: line editing and history in input())

from uuid import uuid4
from pathlib import Path

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from corga import CONF_DIR
from corga.app import repl, build_deps, register_resolvers
from corga.agent.graph import build_graph
from corga.agent.session import ChatSession
from corga.agent.tracing import JsonlTracer

load_dotenv()  # before Hydra resolves ${oc.env:...}
register_resolvers()  # before @hydra.main composes the config


@hydra.main(version_base="1.3", config_path=str(CONF_DIR), config_name="chat")
def main(cfg: DictConfig) -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop → finally blocks run, traces close
    hydra_cfg = HydraConfig.get()
    # hydra=debug has no output folder: conversations are not traced.
    sessions_dir = Path(hydra_cfg.runtime.output_dir) / "sessions" if hydra_cfg.output_subdir else None

    print("Loading the dataset and the embedding model...")
    graph, deps = build_graph(), build_deps(cfg)
    print(f"Model: {hydra_cfg.runtime.choices['provider']} {cfg.provider.model} · weather: {deps.weather.label}")

    def new_session() -> ChatSession:
        session_id = uuid4().hex[:8]
        tracer = JsonlTracer(sessions_dir / f"{session_id}.jsonl") if sessions_dir else None
        return ChatSession(graph, deps, tracer, session_id=session_id)

    repl(new_session)


if __name__ == "__main__":
    main()
