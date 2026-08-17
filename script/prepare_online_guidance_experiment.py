#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import load_dataset, save_dataset  # noqa: E402
from app.pipeline.memory_augmentation.context_graph import (  # noqa: E402
    ContextGraphMemoryAugmentor,
)
from app.prompt import PromptFactory  # noqa: E402


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value)!r}")


def _render_toml_table(lines: list[str], name: str, values: dict[str, Any]) -> None:
    scalars = [(key, value) for key, value in values.items() if not isinstance(value, dict)]
    children = [(key, value) for key, value in values.items() if isinstance(value, dict)]
    if name:
        lines.append(f"[{name}]")
    for key, value in scalars:
        if value is not None:
            lines.append(f"{key} = {_toml_literal(value)}")
    if name or scalars:
        lines.append("")
    for key, value in children:
        child_name = f"{name}.{key}" if name else key
        _render_toml_table(lines, child_name, value)


def dump_toml(values: dict[str, Any]) -> str:
    lines: list[str] = []
    _render_toml_table(lines, "", values)
    return "\n".join(lines).rstrip() + "\n"


def build_experiment_configs(
    base: dict[str, Any],
    output_root: Path,
    schema_path: Path,
    guidance_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    control = copy.deepcopy(base)
    treatment = copy.deepcopy(base)
    for config in (control, treatment):
        config["schema_linking"]["save_path"] = str(schema_path)

    control_memory = control.setdefault("memory_augmentation", {})
    control_memory.update(
        {
            "enabled": False,
            "strategies": [],
            "save_path": str(output_root / "control" / "memory_augmentation.pkl"),
        }
    )

    treatment_memory = treatment.setdefault("memory_augmentation", {})
    treatment_memory.update(
        {
            "enabled": True,
            "strategies": ["context_graph"],
            "use_in_generation": True,
            "use_in_revision": True,
            "use_in_selection": True,
            "save_path": str(output_root / "treatment" / "memory_augmentation.pkl"),
            "context_graph": {
                "mapping_decisions_path": str(guidance_path),
                "format_type": "online_guidance",
                "log_formatted_blocks": False,
            },
        }
    )

    for arm, config in (("control", control), ("treatment", treatment)):
        arm_root = output_root / arm
        config["sql_generation"]["save_path"] = str(arm_root / "sql_generation.pkl")
        config["sql_revision"]["save_path"] = str(arm_root / "sql_revision.pkl")
        config["sql_selection"]["save_path"] = str(arm_root / "sql_selection.pkl")
    return control, treatment


def count_schema_augmented(items: Any) -> int:
    return sum(bool(getattr(item, "memory_schema_overrides", None)) for item in items)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config/config_bird.toml")
    parser.add_argument(
        "--schema-artifact",
        default="workspace/schema_linking/bird/sub_dev_school_and_card_latter50.pkl",
    )
    parser.add_argument("--guidance", required=True)
    parser.add_argument(
        "--output-root",
        default="workspace/experiments/bird/dev/california_schools_44_88_online_guidance",
    )
    parser.add_argument("--database-id", default="california_schools")
    parser.add_argument("--case-start", type=int, default=44)
    parser.add_argument("--case-end", type=int, default=88)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    guidance_source = Path(args.guidance).resolve()
    guidance_copy = output_root / "online_guidance.json"
    shutil.copy2(guidance_source, guidance_copy)

    records = json.loads(guidance_copy.read_text(encoding="utf-8"))
    expected_ids = list(range(args.case_start, args.case_end + 1))
    actual_ids = [
        record["case_index"]
        for record in records
        if record.get("database_id") == args.database_id
    ]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected guidance cases {expected_ids}, got {actual_ids}")

    dataset = load_dataset(args.schema_artifact)
    dataset._data = [
        item
        for item in dataset
        if item.database_id == args.database_id
        and args.case_start <= item.question_id <= args.case_end
    ]
    dataset_ids = [item.question_id for item in dataset]
    if dataset_ids != expected_ids:
        raise ValueError(f"Expected dataset cases {expected_ids}, got {dataset_ids}")

    schema_path = output_root / "schema_linking.pkl"
    save_dataset(dataset, str(schema_path))
    augmentor = ContextGraphMemoryAugmentor(
        {
            "mapping_decisions_path": str(guidance_copy),
            "format_type": "online_guidance",
            "log_formatted_blocks": False,
        }
    )
    for item in dataset:
        augmentor.augment(item)

    treatment_memory_path = output_root / "treatment" / "memory_augmentation.pkl"
    treatment_memory_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, str(treatment_memory_path))

    base_config_path = Path(args.base_config)
    with open(base_config_path, "rb") as f:
        base_config = tomllib.load(f)
    control_config, treatment_config = build_experiment_configs(
        base_config, output_root, schema_path, guidance_copy
    )
    (output_root / "control.toml").write_text(dump_toml(control_config), encoding="utf-8")
    (output_root / "treatment.toml").write_text(
        dump_toml(treatment_config), encoding="utf-8"
    )

    snapshots_dir = output_root / "prompt_snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    snapshot_ids = [args.case_start, 67, args.case_end]
    for item in dataset:
        if item.question_id not in snapshot_ids:
            continue
        hint = PromptFactory.get_sql_generation_hint(
            item, use_caf_mapping=True, use_memory=True
        )
        (snapshots_dir / f"case_{item.question_id}.txt").write_text(
            hint, encoding="utf-8"
        )

    status_counts: dict[str, int] = {}
    for record in records:
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    report = {
        "database_id": args.database_id,
        "case_ids": expected_ids,
        "case_count": len(dataset),
        "guidance_sha256": hashlib.sha256(guidance_copy.read_bytes()).hexdigest(),
        "status_counts": status_counts,
        "resolved_guidance_count": sum(
            bool(item.resolved_user_guidance) for item in dataset
        ),
        "no_guidance_count": sum(
            not bool(item.resolved_user_guidance) for item in dataset
        ),
        "schema_augmented_count": count_schema_augmented(dataset),
        "prompt_snapshot_ids": snapshot_ids,
        "control_config": str(output_root / "control.toml"),
        "treatment_config": str(output_root / "treatment.toml"),
    }
    (output_root / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
