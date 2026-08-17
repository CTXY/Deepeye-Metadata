import subprocess
import sys
import unittest


class PipelineLazyImportTest(unittest.TestCase):
    def test_pipeline_package_does_not_import_optional_stage_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import app.pipeline; print('ok')"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_memory_runner_does_not_import_dependencies_for_disabled_strategies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from app.pipeline import MemoryAugmentationRunner; print(MemoryAugmentationRunner.__name__)",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "MemoryAugmentationRunner")


if __name__ == "__main__":
    unittest.main()
