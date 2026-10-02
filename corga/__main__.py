import sys
import signal
import readline  # noqa: F401  (line editing and history for input())

from uuid import uuid4
from pathlib import Path

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from corga import CONF_DIR
from corga.app import Repl, build_deps, register_resolvers
from corga.agent.runtime import JsonTracer, ChatSession, build_graph

load_dotenv()  # before Hydra resolves ${oc.env:...}
register_resolvers()  # before @hydra.main composes the config


@hydra.main(version_base="1.3", config_path=str(CONF_DIR), config_name="chat")
def main(cfg: DictConfig) -> None:
    """Run the terminal chat with the composed config.

    Args:
        cfg: The config Hydra composed.
    """
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # docker stop runs the finally blocks and closes the traces

    hydra_cfg = HydraConfig.get()
    # hydra=debug has no output folder: conversations are not traced.
    sessions_dir = Path(hydra_cfg.runtime.output_dir) / "sessions" if hydra_cfg.output_subdir else None

    print("Loading the dataset and the embedding model...")
    graph, deps = build_graph(), build_deps(cfg)
    weather = deps.weather.label
    print(f"Model: {hydra_cfg.runtime.choices['provider']} {cfg.provider.model} · weather: {weather}")

    def new_session() -> ChatSession:
        """Start a chat session, traced to its own file unless tracing is off.

        Returns:
            The new session.
        """
        session_id = uuid4().hex[:8]
        tracer = JsonTracer(sessions_dir / f"{session_id}.json") if sessions_dir else None

        return ChatSession(graph, deps, tracer, session_id=session_id)

    Repl(new_session).run()


if __name__ == "__main__":
    main()
