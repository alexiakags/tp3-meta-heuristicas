import tempfile
import unittest
from pathlib import Path

from src.experiments import run_experiments


class ExperimentTests(unittest.TestCase):
    def test_run_experiments_collects_30_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_experiments(runs=30, output_dir=Path(tmp_dir), seed_base=1)
            self.assertIn("tp1_problema1_config_a", result)
            self.assertIn("tp2_problema1_com_restricao", result)
            for problem in result.values():
                self.assertEqual(30, len(problem["runs"]))
                self.assertIn("min", problem["stats"])
                self.assertIn("max", problem["stats"])
                self.assertIn("mean", problem["stats"])
                self.assertIn("std", problem["stats"])

    def test_outputs_summary_and_boxplots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_experiments(runs=5, output_dir=root, seed_base=10)
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "boxplots" / "tp1_problema1_config_a.svg").exists())
            self.assertTrue((root / "boxplots" / "tp2_problema1_com_restricao.svg").exists())


if __name__ == "__main__":
    unittest.main()
