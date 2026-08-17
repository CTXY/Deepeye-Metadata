import unittest
from types import SimpleNamespace

from app.prompt.factory import PromptFactory


class ResolvedUserGuidancePromptTest(unittest.TestCase):
    def setUp(self):
        self.item = SimpleNamespace(
            evidence="Original evidence.",
            resolved_user_guidance=(
                "## Memory guidance for SQL generation\n"
                "- Represent city with `schools.City`."
            ),
            mapping_hint=None,
            mapping_historical_qa=None,
            guidance_hint="advisory guidance must not replace the resolved decision",
            memory_summary=None,
        )

    def test_resolved_guidance_is_labeled_as_highest_priority(self):
        hint = PromptFactory.get_sql_generation_hint(
            self.item,
            use_caf_mapping=True,
            use_memory=True,
        )

        self.assertLess(hint.index("Original evidence."), hint.index("Resolved user-interaction"))
        self.assertIn("highest priority", hint)
        self.assertIn("Represent city with `schools.City`", hint)
        self.assertNotIn("advisory guidance must not replace", hint)

    def test_selection_prompt_uses_resolved_guidance_as_a_decision_criterion(self):
        prompt = PromptFactory.format_br_pair_selection_prompt(
            "schema",
            "question",
            "## Resolved user-interaction guidance\nUse schools.City",
            "SELECT 1",
            "1",
            "SELECT 2",
            "2",
        )

        self.assertIn("resolved user-interaction guidance", prompt.lower())
        self.assertIn("decision criterion", prompt.lower())

    def test_common_revision_prompt_applies_guidance_within_checker_scope(self):
        prompt = PromptFactory.format_common_checker_prompt(
            "schema",
            "question",
            "## Resolved user-interaction guidance\nUse schools.City",
            "SELECT 1",
            "Fix selected columns",
        )

        self.assertIn("resolved user-interaction guidance", prompt.lower())
        self.assertIn("within the scope", prompt.lower())


if __name__ == "__main__":
    unittest.main()
