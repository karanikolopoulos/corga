# CorGA: Corfu Guide Agent

A small conversational travel assistant for Corfu, built with LangGraph. It answers questions about the destination,
recommends places based on the visitor's preferences, produces a realistic half-day plan, and revises that plan when the
visitor changes their mind or the weather changes.

## Setup

```bash
cp .env.example .env
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | the `anthropic` (default) and `openai` providers |
| `OAI_API_BASE_URL`, `OAI_API_KEY` | the `oai-api` provider, any OpenAI-compatible server |
| `QDRANT_URL`, `QDRANT_API_KEY` | a Qdrant server instead of the local index |

The place index is kept:

- on a Qdrant server when `QDRANT_URL` is set (Qdrant Cloud or your own cluster),
- else on disk in `.cache/embeddings`, reused between runs,
- else in memory with `retrieval.qdrant_path=null`, rebuilt on each start.

## Run with Docker

```bash
docker compose build
docker compose run --rm app
```

Options go after `app`, e.g. `docker compose run --rm app provider=openai weather=fixture` (see
[Configuration](#configuration)).
Or `uv run poe docker`, which also builds the image and sets your user and group ids.

## Install locally

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
uv run poe chat
```

With pip (Python 3.12+):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .
python -m corga
```

## Configuration

Configuration and logging are made through [Hydra](https://hydra.cc) inside [`conf/chat.yaml`](conf/chat.yaml).

| Group | Options |
|---|---|
| `provider` | `anthropic` (default), `openai`, `oai-api` |
| `weather` | `live` (default, Open-Meteo), `fixture` (synthetic rain) |
| `embedding` | `minilm` |
| `store` | `poi` |
| `retrieval` | `qdrant` |

Automated tasks run with [poe](https://poethepoet.natn.io), defined in [`tasks.toml`](tasks.toml) (run with
`uv run poe <task>`):

```text
  clean                 Remove all cache and generated files
  lint                  Run ruff linter with diff (pass directory/file as arg)
  format                Check formatting with diff (pass directory/file as arg)
  quality               Run all pre-commit hooks on all files
  test                  Run the unit test suite (pass extra pytest args after --)
  chat                  Start the interactive chat (Hydra overrides go after it)
  docker                Start the chat in Docker as your own user, building the image when needed
```

Choose the provider, and optionally its model:

```bash
uv run poe chat                                             # Anthropic (default model claude-sonnet-5-5)
uv run poe chat +model=claude-opus-5-5                      # Anthropic, another model
uv run poe chat provider=openai                             # OpenAI (default model gpt-5.4-mini)
uv run poe chat provider=openai +model=gpt-5.5              # OpenAI, another model
uv run poe chat provider=oai-api +model=Qwen/Qwen3.8-27B    # an OpenAI-compatible server, see OAI_API_BASE_URL
```

Choose the weather:

```bash
uv run poe chat                                             # the live Open-Meteo forecast (default)
uv run poe chat weather=fixture                             # the synthetic rain forecast
```

Other options:

```bash
uv run poe chat +max_stops=4 +speed_kmh=4                   # planning rules, see conf/rules.yaml
uv run poe chat retrieval.qdrant_path=null                  # keep the place index in memory
uv run poe chat hydra=debug                                 # disable logging, files and storage
```

The same options work with `uv run poe docker` and `docker compose run --rm app`.

Each run writes `runs/<provider>/<model>/<time>/`: the session traces (`sessions/<id>.json`), the log (at DEBUG) and the
resolved configuration (`.hydra/`).

## Tests

```bash
uv run poe test
```

The tests run the graph and the planner with a scripted LLM, the rain fixture and a fixed clock, so they need no API
key and no network.

## Documentation

- [docs/design.md](docs/design.md), how the system separates retrieval, live tools, LLM reasoning, itinerary validation
  and conversation state, and how it can be extended.
- [docs/evaluation.md](docs/evaluation.md), nine conversations with the expected behaviour and the observed result.
- [docs/discussion.md](docs/discussion.md), retrieval quality, the iterative design process and hallucinations.
- [notebooks](notebooks/), the data exploration of the places
  ([1.0-ck-poi-exploration.ipynb](notebooks/1.0-ck-poi-exploration.ipynb)) and of the walking times
  ([1.0-ck-walking-times.ipynb](notebooks/1.0-ck-walking-times.ipynb)).
