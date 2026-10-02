"""The application layer: builds the objects from the Hydra config and runs the terminal chat."""

from corga.app.cli import repl
from corga.app.bootstrap import sanitize, build_deps, register_resolvers

__all__ = [
    "build_deps",
    "register_resolvers",
    "repl",
    "sanitize",
]
