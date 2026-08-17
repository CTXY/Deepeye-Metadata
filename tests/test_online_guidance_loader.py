import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


_LOADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "pipeline"
    / "memory_augmentation"
    / "context_graph_offline_loader.py"
)
_SPEC = importlib.util.spec_from_file_location("context_graph_offline_loader", _LOADER_PATH)
assert _SPEC and _SPEC.loader
_LOADER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LOADER)
ensure_mapping_data = _LOADER.ensure_mapping_data
get_offline_memory_record = _LOADER.get_offline_memory_record


def _schema() -> dict:
    return {
        "tables": {
            "schools": {
                "columns": {
                    "CDSCode": {"column_name": "CDSCode"},
                    "City": {"column_name": "City"},
                }
            },
            "satscores": {
                "columns": {
                    "cds": {"column_name": "cds"},
                    "AvgScrWrite": {"column_name": "AvgScrWrite"},
                }
            },
        }
    }


def _item(question: str = "Which city has the best writing score?") -> SimpleNamespace:
    full_schema = _schema()
    linked_schema = {
        "tables": {
            "schools": {
                "columns": {"CDSCode": {"column_name": "CDSCode"}}
            }
        }
    }
    return SimpleNamespace(
        database_id="california_schools",
        question_id=44,
        question=question,
        database_schema=full_schema,
        database_schema_after_value_retrieval=full_schema,
        database_schema_after_schema_linking=linked_schema,
        mapping_hint="stale mapping",
        mapping_historical_qa=[{"qa_id": "stale"}],
        guidance_hint="stale advisory guidance",
        resolved_user_guidance=None,
    )


class OnlineGuidanceLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def _write_export(
        self,
        *,
        status: str = "GUIDANCE",
        question: str = "Which city has the best writing score?",
    ) -> Path:
        record = {
            "database_id": "california_schools",
            "case_index": 44,
            "question": question,
            "evidence": "",
            "status": status,
            "graph_revision": 65,
            "guidance": [],
            "downstream_guidance": "",
        }
        if status == "GUIDANCE":
            record["guidance"] = [
                {
                    "role_id": "role:city",
                    "candidate_key": "candidate:city",
                    "message": "Use `schools.City` and `satscores.AvgScrWrite`.",
                    "cited_query_ids": ["bird:california_schools:1"],
                },
                {
                    "role_id": "role:join-only",
                    "candidate_key": "none_of_these",
                    "message": "Do not use `schools.CDSCode` as a semantic expression.",
                    "cited_query_ids": [],
                },
            ]
            record["downstream_guidance"] = (
                "## Memory guidance for SQL generation\nUse the reviewed decisions."
            )
        else:
            record["downstream_guidance"] = (
                "## Memory guidance for SQL generation\nGeneric boilerplate."
            )
        path = Path(self.temp_dir.name) / f"online-{status}.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        return path

    def test_array_is_indexed_by_database_and_case(self):
        path = self._write_export()

        record = get_offline_memory_record("california_schools", 44, str(path))

        self.assertIsNotNone(record)
        self.assertEqual(record["case_index"], 44)

    def test_question_mismatch_is_rejected(self):
        path = self._write_export(question="A different question")

        with self.assertRaisesRegex(ValueError, "question mismatch"):
            ensure_mapping_data(_item(), str(path), format_type="online_guidance")

    def test_no_guidance_clears_prompt_despite_boilerplate(self):
        path = self._write_export(status="NO_GUIDANCE")
        item = _item()

        ensure_mapping_data(item, str(path), format_type="online_guidance")

        self.assertIsNone(item.resolved_user_guidance)
        self.assertIsNone(item.guidance_hint)
        self.assertIsNone(item.mapping_hint)
        self.assertIsNone(item.mapping_historical_qa)
        self.assertIsNone(item.database_schema_after_mapping)

    def test_selected_qualified_columns_augment_schema(self):
        path = self._write_export()
        item = _item()

        ensure_mapping_data(item, str(path), format_type="online_guidance")

        self.assertTrue(item.resolved_user_guidance.startswith("## Memory guidance"))
        self.assertEqual(
            item.mapping_linked_tables_and_columns,
            {"schools": ["City"], "satscores": ["AvgScrWrite"]},
        )
        augmented = item.database_schema_after_mapping["tables"]
        self.assertEqual(set(augmented["schools"]["columns"]), {"CDSCode", "City"})
        self.assertEqual(
            set(augmented["satscores"]["columns"]), {"cds", "AvgScrWrite"}
        )


if __name__ == "__main__":
    unittest.main()
