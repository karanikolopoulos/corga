from pathlib import Path

from jinja2 import Template, Environment, StrictUndefined, FileSystemLoader

from corga import TEMPLATES_DIR

FILTERS = {
    "hhmm": lambda t: f"{t:%H:%M}",
    "day": lambda t: f"{t:%A %d %B %Y}",
    "weekday_hour": lambda t: f"{t:%a %H:%M}",
}


class PromptStore:
    """The templates in one folder, by name: `prompts["route"]` is `route.j2`."""

    def __init__(self, directory: Path = TEMPLATES_DIR):
        self.env = Environment(
            loader=FileSystemLoader(directory),
            trim_blocks=True,  # a line holding only {% ... %} leaves no empty line behind
            lstrip_blocks=True,
            undefined=StrictUndefined,  # a missing value raises an error
            autoescape=False,  # markdown and plain text, not HTML
        )
        self.env.filters.update(FILTERS)

    @property
    def names(self) -> list[str]:
        return [name.removesuffix(".j2") for name in self.env.list_templates(extensions=["j2"])]

    def __getitem__(self, name: str) -> Template:
        if name not in self.names:
            raise KeyError(f"no template {name!r}; available: {', '.join(self.names)}")

        return self.env.get_template(f"{name}.j2")
