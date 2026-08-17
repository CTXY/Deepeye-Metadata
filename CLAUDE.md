# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepEye-SQL is a Text-to-SQL framework modeled on the Software Development Life Cycle (SDLC). It was previously named Symph-SQL. It uses ~30B open-source LLMs (no fine-tuning) to achieve 73.5% execution accuracy on BIRD-Dev and 89.8% on Spider-Test.

## Setup & Installation

```bash
# Install dependencies (uses uv package manager)
bash script/install_env.sh
# or manually:
uv sync

# Download datasets (BIRD dev, Spider test)
bash script/download_dataset.sh

# Copy and edit config
cp config/config-example.toml config/config.toml
# Edit config/config.toml: set base_url, api_key, model for each stage's LLM
```

## Running the Pipeline

All stages are run sequentially via `uv run runner/<script>.py`. Each stage reads from the previous stage's `save_path` and writes to its own:

```bash
uv run runner/preprocess_dataset.py          # 1. Preprocess → workspace/dataset/
bash script/make_vector_db.sh                 # 2. Build vector DB → workspace/vector_database/
uv run runner/run_value_retrieval.py          # 3. Value retrieval
uv run runner/run_schema_linking.py           # 4. Schema linking
uv run runner/run_memory_augmentation.py      # 5. Memory augmentation (optional, if enabled)
uv run runner/run_sql_generation.py           # 6. SQL generation (3 strategies in parallel)
uv run runner/run_sql_revision.py             # 7. SQL revision (rule-based + LLM checkers)
uv run runner/run_sql_selection.py            # 8. SQL selection (confidence-aware)
uv run runner/convert_pkl_to_sql_file.py      # 9. Convert pkl → readable JSON
uv run runner/evaluation.py                   # 10. Evaluate against ground truth
```

Each runner supports resumption: if its `save_path` already exists, it loads from there and skips already-processed items.

## Architecture

### Data Flow

The central data structure is `DataItem` (`app/dataset/dataset.py`), a Pydantic model that accumulates fields through each pipeline stage. It is persisted as pickle files (`.pkl`) between stages. Each runner reads the previous stage's `.pkl`, enriches `DataItem` fields, and saves a new `.pkl`.

### Pipeline Stages (`app/pipeline/`)

| Stage | Runner class | Key output field on `DataItem` |
|---|---|---|
| Value Retrieval | `ValueRetrievalRunner` | `retrieved_values`, `database_schema_after_value_retrieval` |
| Schema Linking | `SchemaLinkingRunner` | `final_linked_tables_and_columns`, `database_schema_after_schema_linking` |
| Memory Augmentation | `MemoryAugmentationRunner` | `memory_summary`, `memory_schema_overrides` |
| SQL Generation | `SQLGenerationRunner` | `sql_candidates` |
| SQL Revision | `SQLRevisionRunner` | `sql_candidates_after_revision` |
| SQL Selection | `BRSelectionRunner` | `final_selected_sql` |

### SQL Generation — N-Version Strategy

`SQLGenerationRunner` uses three generators in parallel (all configurable with `*_sampling_budget`):
- **DCGenerator** (`dc_generator.py`): Direct chain-of-thought prompting
- **SkeletonGenerator** (`skeleton_generator.py`): First generates a SQL skeleton, then fills it
- **ICLGenerator** (`icl_generator.py`): In-context learning with few-shot examples (`icl_few_shot_examples_path`)

### SQL Revision — Checker Chain

`SQLRevisionRunner` applies multiple rule-based and LLM-guided checkers from `app/pipeline/sql_revision/checkers/`. Each checker implements `BaseChecker.check_and_revise(sql, data_item, llm, sampling_budget)`. Checkers include: `SyntaxChecker`, `ResultChecker` (catches execution errors), `JoinChecker`, `SelectChecker`, `MaxMinChecker`, `OrderByLimitChecker`, `OrderByNullChecker`.

### SQL Selection — Confidence-Aware

`BRSelectionRunner` clusters SQL candidates by execution result and computes a consistency score. If top-1 exceeds `shortcut_consistency_score_threshold`, it is returned directly. Otherwise, unbalanced pairwise LLM adjudication (A/B/TIE votes) determines the winner.

### Schema Linking — Three Linkers

`SchemaLinkingRunner` merges results from three linkers:
- **DirectLinker**: LLM identifies tables/columns from the question
- **ReversedLinker**: LLM drafts a preliminary SQL, then extracts schema used
- **ValueLinker**: Vector-DB lookup matches question values to columns

### Configuration (`app/config/config.py`)

A singleton `Config` class loaded from `config/config.toml` (TOML format). Each pipeline stage has its own config section with an `llm` sub-section. Access via `from app.config import config`, then e.g. `config.sql_generation_config.dc_sampling_budget`.

### LLM Interface (`app/llm/llm.py`)

`LLM` is a singleton per model name; supports `api_type = "openai"` or `"azure"`. The `ask()` method wraps the OpenAI client with tenacity retry on `RateLimitError`. All stages use `stop=["</result>"]` and parse the `<result>...</result>` tag from responses.

### Prompt System (`app/prompt/`)

`PromptFactory` in `app/prompt/factory.py` centralizes prompt construction. It selects the correct schema (with optional memory overrides) and hint string for each stage, controlled by `memory_augmentation.use_in_generation/revision/selection` flags.

### Memory Augmentation

Optional stage controlled by `memory_augmentation.enabled` in config. Strategies: `context_graph` (uses pre-built mapping decisions JSONL), `historical_qa` (vector retrieval from past QA), `amem_retrieval` (pre-retrieved A-mem notes). Enriches `DataItem.memory_summary` and `memory_schema_overrides` used by downstream prompts.

**重要：memory 数据是 pkl 快照，不会自动感知 JSONL 更新。**

每个阶段的 pkl 文件在生成时会把 memory context（`memory_items`、`mapping_historical_qa` 等）**直接 embed 进 `DataItem`**。因此：

- `workspace/sql_generation/...pkl` 里的 memory 数据是 sql_generation 那次跑时的快照。
- 即使事后更新了 `offline_memory_augmentation.jsonl` 并重跑了 `run_memory_augmentation.py`，`--start-from sql_generation` 仍然加载旧的 sql_generation pkl，**memory 数据不会被刷新**。
- sql_selection 阶段用 memory context 做候选 SQL 的 LLM 投票，stale memory 会导致错误的 SQL 被选中（即使 sql_candidates 里有正确答案）。

**正确流程：更新 JSONL 后必须从 memory_augmentation 或更早阶段重新开始。**

```
# 正确：memory_augmentation pkl 是新跑的，包含最新 memory 数据
python runner/run_memory_augmentation.py
nohup python runner/run_stream_pipeline.py --start-from memory_augmentation ...

# 错误：sql_generation pkl 里的 memory 是旧快照，不会被刷新
python runner/run_memory_augmentation.py
nohup python runner/run_stream_pipeline.py --start-from sql_generation ...  # ← 仍用旧 memory
```

`--start-from X` 对应加载的 pkl 和执行的剩余阶段：

| `--start-from` | 加载的 pkl | 剩余阶段 |
|---|---|---|
| `memory_augmentation` | `workspace/memory_augmentation/...pkl` | sql_generation → sql_revision → sql_selection |
| `sql_generation` | `workspace/sql_generation/...pkl` | sql_revision → sql_selection（**memory_augmentation 不重跑**） |

### Vector Database (`app/vector_db/`)

Uses ChromaDB (`PersistentClient`) with per-database collections. Embedding model is configurable; Qwen3-Embedding-0.6B is the default. Set `use_qwen3_embedding = true` for Qwen3-specific prompt formatting.

## Key Configuration Parameters

- `database_whitelist`: Subset of database IDs to run (useful for debugging)
- `n_parallel`: Thread pool size for each stage (reduce for debugging)
- `*_sampling_budget`: Number of LLM samples per item per generator/checker
- `shortcut_consistency_score_threshold`: Below this, SQL selection uses pairwise adjudication
- `filter_top_k_sql`: Number of unique-result candidates passed to pairwise selection
