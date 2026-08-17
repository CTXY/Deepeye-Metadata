# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the core Python package: `config/` for TOML-backed settings, `db_utils/` for schema and execution helpers, `llm/` for model access, `prompt/` for prompt construction, `vector_db/` for retrieval storage, and `pipeline/` for stage implementations such as `schema_linking/`, `sql_generation/`, `sql_revision/`, and `sql_selection/`. Use `runner/` for entry-point scripts, `script/` for environment and dataset setup, `config/` for repository configs, `data/` and `dataset/` for inputs, and `workspace/` for generated `.pkl`, logs, and evaluation artifacts.

## Build, Test, and Development Commands
Install dependencies with `uv sync` or `bash script/install_env.sh`. Download datasets with `bash script/download_dataset.sh`. Run stages individually with `uv run runner/preprocess_dataset.py`, `uv run runner/run_schema_linking.py`, or other `runner/*.py` scripts. Use `bash runner/run_full_pipeline.sh` to execute the later SQL generation, revision, selection, conversion, and evaluation stages in sequence. Configuration is loaded from `config/config.toml`; start by copying the example file and setting model endpoints before running anything.

## Coding Style & Naming Conventions
Target Python 3.12+ and follow the existing style: 4-space indentation, type hints on public functions, and `snake_case` for modules, functions, variables, and config keys. Keep Pydantic models explicit and prefer small stage-specific classes over monolithic scripts. There is no enforced formatter or linter in the repo today, so match surrounding code closely and keep imports, logging, and docstrings consistent with nearby files.

## Testing Guidelines
There is no dedicated `tests/` directory yet. Treat pipeline runs as the current validation path: execute the smallest affected runner, then verify outputs under `workspace/` and, when relevant, run `uv run runner/evaluation.py`. For new test coverage, add focused `test_*.py` files near the relevant module or introduce a top-level `tests/` package with lightweight fixtures instead of full dataset copies.

## Commit & Pull Request Guidelines
Git history is minimal and uses short, imperative summaries such as `Add caf/memory directory...` and `Initial commit...`. Continue with concise subject lines that describe the behavioral change. Pull requests should explain the affected pipeline stage, required config or dataset assumptions, the commands used for validation, and any result deltas. Include sample output paths or screenshots only when they clarify evaluation changes.

## Configuration & Data Notes
Do not commit secrets, API keys, or large generated artifacts from `workspace/`. Many runners resume from existing `.pkl` files, so when changing memory augmentation or upstream data, rerun from the earliest affected stage rather than relying on downstream caches.
