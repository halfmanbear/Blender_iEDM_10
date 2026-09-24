"""Boundary tests for the incremental Python quality gate."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import check_code_standard as standard


class CodeStandardTests(unittest.TestCase):
    def test_logical_lines_count_multiline_statement_once(self):
        source = (
            "def example():\n"
            "    value = (\n        1 +\n        2\n    )\n"
            "    return value\n"
        )
        self.assertEqual(standard.function_lines(source)["example"], 3)

    def test_new_file_limit_and_legacy_no_growth(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "large.py"
            baseline = {
                "legacy_files": {"large.py": 401},
                "legacy_functions": {},
                "ruff_findings": {},
            }
            with (
                patch.object(standard, "ROOT", root),
                patch.object(standard, "ruff_findings", return_value={}),
            ):
                path.write_text("# note\n" * 401)
                self.assertEqual(standard.check_baseline(baseline, [path]), [])
                path.write_text("# note\n" * 402)
                self.assertIn(
                    "402 physical lines", standard.check_baseline(baseline, [path])[0]
                )
                self.assertIn(
                    "402 physical lines",
                    standard.check_baseline({**baseline, "legacy_files": {}}, [path])[
                        0
                    ],
                )
                path.write_text("# note\n" * 399)
                self.assertIn("obsolete", standard.check_baseline(baseline, [path])[0])
                self.assertEqual(
                    standard.check_baseline({**baseline, "legacy_files": {}}, [path]),
                    [],
                )

    def test_legacy_function_may_shrink_but_not_grow(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.py"
            baseline = {
                "legacy_files": {},
                "legacy_functions": {"legacy.py": {"legacy": 102}},
                "ruff_findings": {},
            }
            with (
                patch.object(standard, "ROOT", root),
                patch.object(standard, "ruff_findings", return_value={}),
            ):
                path.write_text("def legacy():\n" + "    pass\n" * 101)
                self.assertEqual(standard.check_baseline(baseline, [path]), [])
                path.write_text("def legacy():\n" + "    pass\n" * 102)
                self.assertTrue(standard.check_baseline(baseline, [path]))

    def test_new_function_and_ruff_finding_fail(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "new.py"
            path.write_text("def over_limit():\n" + "    pass\n" * 101)
            baseline = {
                "legacy_files": {},
                "legacy_functions": {},
                "ruff_findings": {},
            }
            with (
                patch.object(standard, "ROOT", root),
                patch.object(
                    standard, "ruff_findings", return_value={"new.py": {"F401": 1}}
                ),
            ):
                errors = standard.check_baseline(baseline, [path])
            self.assertTrue(any("over_limit" in error for error in errors))
            self.assertTrue(any("F401" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
