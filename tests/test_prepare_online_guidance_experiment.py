import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


_SCRIPT = Path(__file__).resolve().parents[1] / "script" / "prepare_online_guidance_experiment.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_online_guidance_experiment", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareOnlineGuidanceExperimentTest(unittest.TestCase):
    def test_schema_augmentation_count_supports_older_pickles_without_field(self):
        module = _load_script()

        count = module.count_schema_augmented(
            [SimpleNamespace(memory_schema_overrides={"tables": {}}), SimpleNamespace()]
        )

        self.assertEqual(count, 1)

    def test_control_and_treatment_share_model_settings_but_isolate_memory_and_outputs(self):
        module = _load_script()
        base = {
            "schema_linking": {"save_path": "old-schema.pkl"},
            "memory_augmentation": {
                "enabled": True,
                "strategies": ["historical_qa"],
                "use_in_generation": True,
                "use_in_revision": True,
                "use_in_selection": True,
            },
            "sql_generation": {"save_path": "old-generation.pkl", "n_parallel": 2},
            "sql_revision": {"save_path": "old-revision.pkl", "n_parallel": 2},
            "sql_selection": {"save_path": "old-selection.pkl", "n_parallel": 2},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            control, treatment = module.build_experiment_configs(
                base,
                root,
                root / "schema_linking.pkl",
                root / "online_guidance.json",
            )

        self.assertFalse(control["memory_augmentation"]["enabled"])
        self.assertEqual(control["memory_augmentation"]["strategies"], [])
        self.assertTrue(treatment["memory_augmentation"]["enabled"])
        self.assertEqual(treatment["memory_augmentation"]["strategies"], ["context_graph"])
        self.assertEqual(
            treatment["memory_augmentation"]["context_graph"]["format_type"],
            "online_guidance",
        )
        self.assertTrue(treatment["memory_augmentation"]["use_in_generation"])
        self.assertTrue(treatment["memory_augmentation"]["use_in_revision"])
        self.assertTrue(treatment["memory_augmentation"]["use_in_selection"])
        self.assertEqual(control["sql_generation"]["n_parallel"], treatment["sql_generation"]["n_parallel"])
        self.assertNotEqual(
            control["sql_generation"]["save_path"],
            treatment["sql_generation"]["save_path"],
        )


if __name__ == "__main__":
    unittest.main()
