# Contributing to Edward

Thank you for your interest in contributing to Edward!

## Development Philosophy

1. **Capture First:** The original capture must always commit before network enrichment, classification, extraction, or embeddings.
2. **Preserve Provenance:** Always record where material came from, who added it, when it entered, and exact locators for supporting claims.
3. **Model as Derived Data:** Model outputs are versioned, reviewable, rebuildable, and start in the `unreviewed` state.
4. **Agent Neutrality:** External agents and human users interact through identical public CLI and JSON interfaces.
5. **Clean Repository:** Never commit private corpus data, credentials, real personal bookmarks, or machine-specific absolute paths. Tests use synthetic fixtures.

## Getting Started

1. Install [`uv`](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone repository and install dependencies:
   ```bash
   git clone https://github.com/nmorton/edward.git
   cd edward
   uv sync --all-groups
   ```
3. Run tests:
   ```bash
   uv run pytest
   ```

## Code Quality & Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:
```bash
uv run ruff check .
uv run ruff format --check .

# Automatically apply formatting
uv run ruff format .
uv run ruff check --fix .
```

## Pull Request Guidelines

- Ensure all existing and new unit tests pass with `uv run pytest`.
- Ensure packaging builds cleanly with `uv build`.
- Avoid committing credentials, personal tokens, or real user emails.
