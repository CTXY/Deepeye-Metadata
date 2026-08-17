#!/usr/bin/env python3
"""
Materialize a tail-percentage experiment subset from a full upstream stage artifact.

Typical workflow:
  1. Run expensive full stages once using a base config:
     preprocess -> vector_db -> value_retrieval -> schema_linking
  2. Create a subset experiment from the full schema_linking artifact:
     python script/materialize_stage_subset.py \
       --base-config config/config_bird.toml \
       --source-stage schema_linking \
       --tail-percent 50 \
       --experiment-name latter50
  3. Continue downstream stages with the emitted overlay config:
     python runner/run_memory_augmentation.py --config workspace/experiments/bird/dev/latter50/overlay.toml
"""

import argparse
import json
import math
import os
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ENV_VAR = "DEEPEYE_CONFIG_PATH"


def _configure_base_config_from_argv() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-config", type=str)
    args, _ = parser.parse_known_args()
    if args.base_config:
        os.environ[CONFIG_ENV_VAR] = args.base_config


_configure_base_config_from_argv()
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import load_dataset, save_dataset  # noqa: E402
from app.logger import logger  # noqa: E402


DOWNSTREAM_STAGE_KEYS = [
    "augmented_data_retrieval",
    "mapping_analysis",
    "memory_augmentation",
    "sql_generation",
    "sql_revision",
    "sql_selection",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reusable subset experiment from a full stage artifact.",
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Base TOML config used for the full upstream run.",
    )
    parser.add_argument(
        "--source-stage",
        choices=["schema_linking"],
        default="schema_linking",
        help="Which full-stage artifact to slice. Currently only schema_linking is supported.",
    )
    parser.add_argument(
        "--tail-percent",
        type=float,
        required=True,
        help="Keep the last N percent of items by question_id order.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment directory name, e.g. latter50 or latter30_hqa.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--per-database",
        dest="per_database",
        action="store_true",
        help="Apply tail selection within each database_id separately.",
    )
    group.add_argument(
        "--global-order",
        dest="per_database",
        action="store_false",
        help="Apply tail selection over the whole dataset by global question_id order.",
    )
    parser.set_defaults(per_database=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional experiment root directory. Defaults to workspace/experiments/<type>/<split>/<experiment-name>.",
    )
    return parser.parse_args()


def _validate_percent(percent: float) -> None:
    if not (0 < percent <= 100):
        raise ValueError(f"--tail-percent must be in (0, 100], got {percent}")


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT).as_posix())
    except ValueError:
        return str(path.resolve())


def _slugify(name: str) -> str:
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    result = "".join(allowed).strip("_")
    if not result:
        raise ValueError("experiment-name must contain at least one alphanumeric character")
    return result


def _select_tail_items(data_items: list[Any], tail_percent: float, per_database: bool) -> tuple[list[Any], dict[str, Any]]:
    selected: list[Any] = []
    manifest_groups: dict[str, Any] = {}

    if per_database:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for item in data_items:
            grouped[item.database_id].append(item)

        for db_id in sorted(grouped):
            items = sorted(grouped[db_id], key=lambda item: item.question_id)
            start_idx = int(len(items) * (1.0 - tail_percent / 100.0))
            kept = items[start_idx:]
            selected.extend(kept)
            manifest_groups[db_id] = {
                "total_items": len(items),
                "kept_items": len(kept),
                "selected_question_ids": [item.question_id for item in kept],
                "selected_question_id_range": (
                    [kept[0].question_id, kept[-1].question_id] if kept else None
                ),
            }
    else:
        items = sorted(data_items, key=lambda item: item.question_id)
        start_idx = int(len(items) * (1.0 - tail_percent / 100.0))
        selected = items[start_idx:]
        manifest_groups["__global__"] = {
            "total_items": len(items),
            "kept_items": len(selected),
            "selected_question_ids": [item.question_id for item in selected],
            "selected_question_id_range": (
                [selected[0].question_id, selected[-1].question_id] if selected else None
            ),
        }

    selected = sorted(selected, key=lambda item: (item.database_id, item.question_id))
    return selected, manifest_groups


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return repr(value)
        raise ValueError(f"Cannot serialize non-finite float: {value}")
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value is None:
        raise ValueError("None values are not supported in generated TOML")
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            raise ValueError("List of dicts must be serialized as array-of-tables")
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _render_table(lines: list[str], table_name: str, mapping: dict[str, Any]) -> None:
    scalar_items: list[tuple[str, Any]] = []
    dict_items: list[tuple[str, dict[str, Any]]] = []
    array_table_items: list[tuple[str, list[dict[str, Any]]]] = []

    for key, value in mapping.items():
        if isinstance(value, dict):
            dict_items.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            array_table_items.append((key, value))
        elif value is not None:
            scalar_items.append((key, value))

    if table_name:
        lines.append(f"[{table_name}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_literal(value)}")
    if table_name or scalar_items:
        lines.append("")

    for key, value in array_table_items:
        array_name = f"{table_name}.{key}" if table_name else key
        for item in value:
            lines.append(f"[[{array_name}]]")
            for item_key, item_value in item.items():
                if isinstance(item_value, dict):
                    raise ValueError("Nested dicts inside array-of-tables are not supported")
                if item_value is not None:
                    lines.append(f"{item_key} = {_toml_literal(item_value)}")
            lines.append("")

    for key, value in dict_items:
        child_name = f"{table_name}.{key}" if table_name else key
        _render_table(lines, child_name, value)


def _dump_toml(mapping: dict[str, Any]) -> str:
    lines: list[str] = []
    _render_table(lines, "", mapping)
    return "\n".join(lines).rstrip() + "\n"


def _rewrite_stage_paths(base_config: dict[str, Any], experiment_root: Path) -> dict[str, Any]:
    updated = json.loads(json.dumps(base_config))

    schema_linking_path = experiment_root / "schema_linking.pkl"
    updated.setdefault("schema_linking", {})["save_path"] = _project_relative(schema_linking_path)

    for stage_key in DOWNSTREAM_STAGE_KEYS:
        stage_config = updated.get(stage_key)
        if not isinstance(stage_config, dict):
            continue
        stage_config["save_path"] = _project_relative(experiment_root / f"{stage_key}.pkl")

    return updated


def main() -> None:
    args = _parse_args()
    _validate_percent(args.tail_percent)

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = (PROJECT_ROOT / base_config_path).resolve()
    base_config = _load_toml(base_config_path)

    dataset_cfg = base_config["dataset"]
    experiment_name = _slugify(args.experiment_name)

    if args.output_root:
        experiment_root = Path(args.output_root)
        if not experiment_root.is_absolute():
            experiment_root = (PROJECT_ROOT / experiment_root).resolve()
    else:
        experiment_root = (
            PROJECT_ROOT
            / "workspace"
            / "experiments"
            / str(dataset_cfg["type"])
            / str(dataset_cfg["split"])
            / experiment_name
        ).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    source_stage_cfg = base_config.get(args.source_stage)
    if not isinstance(source_stage_cfg, dict) or "save_path" not in source_stage_cfg:
        raise ValueError(f"Missing {args.source_stage}.save_path in {base_config_path}")

    source_path = Path(source_stage_cfg["save_path"])
    if not source_path.is_absolute():
        source_path = (PROJECT_ROOT / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source stage artifact not found: {source_path}")

    dataset = load_dataset(str(source_path))
    original_items = list(dataset._data)
    selected_items, group_manifest = _select_tail_items(
        original_items,
        tail_percent=args.tail_percent,
        per_database=args.per_database,
    )
    if not selected_items:
        raise ValueError("Selection produced zero items; adjust --tail-percent")

    dataset._data = selected_items
    subset_pkl_path = experiment_root / "schema_linking.pkl"
    save_dataset(dataset, str(subset_pkl_path))

    overlay_config = _rewrite_stage_paths(base_config, experiment_root)
    overlay_config_path = experiment_root / "overlay.toml"
    overlay_config_path.write_text(_dump_toml(overlay_config), encoding="utf-8")

    manifest = {
        "base_config_path": _project_relative(base_config_path),
        "source_stage": args.source_stage,
        "source_stage_artifact": _project_relative(source_path),
        "subset_stage_artifact": _project_relative(subset_pkl_path),
        "overlay_config_path": _project_relative(overlay_config_path),
        "dataset_type": dataset_cfg["type"],
        "dataset_split": dataset_cfg["split"],
        "experiment_name": experiment_name,
        "tail_percent": args.tail_percent,
        "per_database": args.per_database,
        "original_item_count": len(original_items),
        "selected_item_count": len(selected_items),
        "databases": group_manifest,
    }
    manifest_path = experiment_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(
        "Materialized subset experiment '{}': kept {}/{} items",
        experiment_name,
        len(selected_items),
        len(original_items),
    )
    logger.info("Subset schema_linking artifact: {}", subset_pkl_path)
    logger.info("Overlay config: {}", overlay_config_path)
    logger.info("Manifest: {}", manifest_path)
    print(f"Next config: {overlay_config_path}")


if __name__ == "__main__":
    main()
