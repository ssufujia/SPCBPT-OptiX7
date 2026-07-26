import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sweep_cuda_mirror_learning_rate.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sweep_cuda_mirror_learning_rate", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MirrorLearningRateSweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_default_experiment_root_matches_current_generator(self):
        self.assertEqual(
            self.module.parse_args([]).experiment_root,
            Path("build/experiments/optimal-e-multiscene-v4"),
        )

    def test_learning_rates_must_be_unique_positive_finite_values(self):
        self.assertEqual(
            self.module.parse_learning_rates("0.01,0.05,0.1"),
            [0.01, 0.05, 0.1],
        )
        for invalid in ("", "0", "-0.1", "nan", "0.01,0.01"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.module.parse_learning_rates(invalid)

    def test_validator_command_passes_selected_learning_rate(self):
        command = self.module.build_validator_command(
            Path("validator.exe"),
            Path("problem.spcoe"),
            Path("result.spcor"),
            0.05,
        )
        self.assertEqual(
            command,
            [
                "validator.exe",
                "--snapshot=problem.spcoe",
                "--result=result.spcor",
                "--learning-rate=0.05",
            ],
        )

    def test_relative_improvement_rejects_zero_initial_loss(self):
        with self.assertRaises(ValueError):
            self.module.relative_improvement(0.0, 0.0)

    def test_aggregate_ranks_rates_by_same_run_relative_improvement(self):
        rows = [
            {
                "initial_loss": 100.0,
                "rates": {
                    "0.01": {"final_loss": 90.0, "accepted_steps": 20},
                    "0.05": {"final_loss": 80.0, "accepted_steps": 20},
                },
            },
            {
                "initial_loss": 200.0,
                "rates": {
                    "0.01": {"final_loss": 180.0, "accepted_steps": 19},
                    "0.05": {"final_loss": 150.0, "accepted_steps": 20},
                },
            },
        ]

        aggregate = self.module.aggregate_rows(rows, [0.01, 0.05])

        self.assertAlmostEqual(
            aggregate["rates"]["0.01"]["mean_relative_improvement"],
            0.1,
        )
        self.assertAlmostEqual(
            aggregate["rates"]["0.05"]["mean_relative_improvement"],
            0.225,
        )
        self.assertEqual(aggregate["rates"]["0.01"]["win_count"], 0)
        self.assertEqual(aggregate["rates"]["0.05"]["win_count"], 2)
        self.assertAlmostEqual(
            aggregate["rates"]["0.01"]["mean_accepted_steps"],
            19.5,
        )


if __name__ == "__main__":
    unittest.main()
