import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_PATH = _ROOT / "app" / "pipeline"
_MEMORY_PATH = _PIPELINE_PATH / "memory_augmentation"


def _package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_package("app.pipeline", _PIPELINE_PATH)
_package("app.pipeline.memory_augmentation", _MEMORY_PATH)
_load("app.pipeline.memory_augmentation.base", _MEMORY_PATH / "base.py")
_load(
    "app.pipeline.memory_augmentation.context_graph_offline_loader",
    _MEMORY_PATH / "context_graph_offline_loader.py",
)
_CONTEXT = _load(
    "app.pipeline.memory_augmentation.context_graph",
    _MEMORY_PATH / "context_graph.py",
)
ContextGraphMemoryAugmentor = _CONTEXT.ContextGraphMemoryAugmentor


def _item() -> SimpleNamespace:
    full_schema = {
        "tables": {
            "schools": {
                "columns": {
                    "CDSCode": {"column_name": "CDSCode"},
                    "City": {"column_name": "City"},
                }
            }
        }
    }
    return SimpleNamespace(
        database_id="california_schools",
        question_id=44,
        question="Which city?",
        database_schema=full_schema,
        database_schema_after_value_retrieval=full_schema,
        database_schema_after_schema_linking={
            "tables": {
                "schools": {
                    "columns": {"CDSCode": {"column_name": "CDSCode"}}
                }
            }
        },
        mapping_hint=None,
        mapping_historical_qa=None,
        guidance_hint=None,
        resolved_user_guidance=None,
        memory_items=None,
        memory_summary=None,
        memory_metadata=None,
    )


class OnlineGuidanceAugmentorTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "guidance.json"

    def _write(self, records: list[dict]) -> None:
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def test_structured_items_metadata_and_schema_override_are_preserved(self):
        guidance_item = {
            "role_id": "role:city",
            "candidate_key": "candidate:city",
            "message": "Use `schools.City`.",
            "cited_query_ids": ["bird:california_schools:30"],
        }
        self._write(
            [
                {
                    "database_id": "california_schools",
                    "case_index": 44,
                    "question": "Which city?",
                    "evidence": "",
                    "status": "GUIDANCE",
                    "graph_revision": 65,
                    "guidance": [guidance_item],
                    "downstream_guidance": "## Memory guidance for SQL generation\nUse city.",
                }
            ]
        )
        item = _item()
        augmentor = ContextGraphMemoryAugmentor(
            {"mapping_decisions_path": str(self.path), "format_type": "online_guidance"}
        )

        augmentor.augment(item)

        self.assertEqual(
            item.memory_items,
            [{"strategy": "context_graph", "type": "resolved_user_guidance", "rank": 1, **guidance_item}],
        )
        self.assertEqual(item.memory_metadata["context_graph"]["status"], "GUIDANCE")
        self.assertEqual(item.memory_metadata["context_graph"]["graph_revision"], 65)
        self.assertEqual(item.memory_metadata["context_graph"]["n_guidance"], 1)
        self.assertIn("City", item.memory_schema_overrides["tables"]["schools"]["columns"])

    def test_missing_online_case_is_an_error(self):
        self._write([])
        augmentor = ContextGraphMemoryAugmentor(
            {"mapping_decisions_path": str(self.path), "format_type": "online_guidance"}
        )

        with self.assertRaisesRegex(ValueError, "No online guidance record"):
            augmentor.augment(_item())


if __name__ == "__main__":
    unittest.main()
