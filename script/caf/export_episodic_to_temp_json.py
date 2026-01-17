'''
python script/export_episodic_to_temp_json.py --input /home/yangchenyu/DeepEye-SQL-Metadata/workspace/sql_selection/bird/sub_dev_episodic_from_train.pkl --output /home/yangchenyu/DeepEye-SQL-Metadata/workspace/sql_selection/bird/sub_dev_episodic_from_train_check.json
'''
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.append(".")

from app.dataset import load_dataset  # noqa: E402
from app.config import config  # noqa: E402
from app.logger import logger  # noqa: E402


def _build_entry(item) -> Dict[str, Any]:
    """Pick the user-facing fields we care about."""
    return {
        "question_id": item.question_id,
        "question": item.question,
        "database_id": item.database_id,
        "final_selected_sql": item.final_selected_sql,
        "episodic_cases": item.episodic_cases,
        "episodic_hint": item.episodic_hint,
        "combined_hint": item.combined_hint,
    }


def export(input_path: Path, output_path: Path) -> None:
    dataset = load_dataset(str(input_path))
    export_payload = [_build_entry(item) for item in dataset]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(export_payload, fp, ensure_ascii=False, indent=2)
    logger.info(f"Exported {len(export_payload)} entries to {output_path}")


def main() -> None:
    default_input = Path(config.sql_selection_config.save_path)
    parser = argparse.ArgumentParser(
        description="Export episodic details + final SQL from a pickled dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to the serialized dataset (.pkl). "
        "Defaults to config.sql_selection_config.save_path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON snapshot. Defaults to temp.json next to the input.",
    )
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name("temp.json")
    )

    logger.info(f"Loading dataset from {input_path}")
    export(input_path, output_path)


if __name__ == "__main__":
    main()

