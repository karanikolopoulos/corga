"""The dependency rule (spec 4.1): each layer imports only the layers below it and no framework it must not know."""

import ast

from pathlib import Path
from collections.abc import Iterator

from corga import ROOT_DIR

FRAMEWORKS = ("langchain", "langgraph", "httpx", "qdrant_client", "hydra", "omegaconf", "jinja2")
QDRANT = ("qdrant_client", "langchain_qdrant")

# layer → (the corga layers it may import, the prefixes of the packages it must not import); `root` is `corga` itself
RULES: dict[str, tuple[set[str], tuple[str, ...]]] = {
    "domain": ({"domain"}, FRAMEWORKS),
    "services": ({"services", "domain", "tracing"}, FRAMEWORKS),
    "adapters": ({"adapters", "domain", "root"}, ("langgraph", "hydra", "omegaconf", "jinja2")),
    "tracing": ({"tracing", "domain"}, ("langgraph", "httpx", *QDRANT, "hydra", "omegaconf", "jinja2")),
    "agent": ({"agent", "services", "domain", "tracing", "root"}, ("httpx", *QDRANT, "hydra", "omegaconf")),
}


def module_name(path: Path) -> str:
    """`corga/agent/state.py` → `corga.agent.state`; a package's `__init__.py` is the package."""
    return ".".join(path.relative_to(ROOT_DIR).with_suffix("").parts).removesuffix(".__init__")


def layer(module: str) -> str:
    """The package under `corga` a module belongs to; `app` for `corga.__main__`, `root` for `corga` itself."""
    parts = module.split(".")
    if len(parts) == 1:
        return "root"
    return "app" if parts[1] == "__main__" else parts[1]


def imports(path: Path) -> Iterator[str]:
    """Every module the file imports, imports inside functions included."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def breaks_the_rule(module: str, imported: str) -> bool:
    allowed, banned = RULES[layer(module)]
    if imported.startswith("corga"):
        return layer(imported) not in allowed
    return imported.startswith(banned)


def test_each_layer_imports_only_what_the_dependency_rule_allows():
    broken = []
    for path in sorted((ROOT_DIR / "corga").rglob("*.py")):
        module = module_name(path)
        if layer(module) not in RULES:
            continue  # the app layer, `__main__` and the root package may import anything
        broken += [f"{module} imports {imported}" for imported in imports(path) if breaks_the_rule(module, imported)]
    assert broken == []
