import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_optimal_e_optimizers.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_optimal_e_optimizers", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CompareOptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_default_experiment_root_matches_current_generator(self):
        self.assertEqual(
            self.module.parse_args([]).experiment_root,
            Path("build/experiments/optimal-e-multiscene-v4"),
        )

    def test_run_name_matches_multiscene_layout(self):
        run = {
            "index": 7,
            "scene_id": "white_room",
            "camera_index": 2,
            "camera": "yaw_p12",
            "experiment_seed": 47,
        }
        self.assertEqual(
            self.module.run_name(run),
            "007-white_room-c02-yaw_p12-s47",
        )

    def test_aggregate_compares_relative_improvement_per_run(self):
        rows = [
            {
                "scene": "a",
                "initial_loss": 10.0,
                "cuda_mirror_final_loss": 8.0,
                "pytorch_adam_final_loss": 7.0,
                "legacy_adam_original_final_loss": 9.0,
                "legacy_adam_matched_final_loss": 7.5,
            },
            {
                "scene": "a",
                "initial_loss": 100.0,
                "cuda_mirror_final_loss": 90.0,
                "pytorch_adam_final_loss": 80.0,
                "legacy_adam_original_final_loss": 95.0,
                "legacy_adam_matched_final_loss": 85.0,
            },
        ]

        summary = self.module.aggregate_rows(rows)

        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(
            summary["methods"]["pytorch_adam"]["win_count"], 2
        )
        self.assertAlmostEqual(
            summary["methods"]["cuda_mirror"][
                "mean_relative_improvement"
            ],
            0.15,
        )
        self.assertAlmostEqual(
            summary["methods"]["legacy_adam_matched"][
                "mean_relative_improvement"
            ],
            0.2,
        )

    def test_cached_variant_rejects_changed_steps(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            q_path = root / "candidate.f32"
            metrics_path = root / "candidate.json"
            provenance_path = root / "candidate.provenance.json"
            q_path.write_bytes(b"\0" * 16)
            metrics = {
                "schema_version": 1,
                "optimizer": "fixed_legacy_sigmoid_adam_cuda",
                "steps": 20,
                "learning_rate": 0.01,
            }
            expected = {
                "schema_version": 1,
                "optimizer": "fixed_legacy_sigmoid_adam_cuda",
                "steps": 20,
                "learning_rate": 0.01,
                "snapshot_sha256": "snapshot",
                "executable_sha256": "executable",
            }
            self.module.write_json(metrics_path, metrics)
            provenance = dict(expected)
            provenance["q_sha256"] = self.module.file_sha256(q_path)
            self.module.write_json(provenance_path, provenance)

            self.assertIsNotNone(
                self.module.load_cached_legacy_variant(
                    q_path,
                    metrics_path,
                    provenance_path,
                    expected,
                )
            )
            changed = dict(expected)
            changed["steps"] = 40
            self.assertIsNone(
                self.module.load_cached_legacy_variant(
                    q_path,
                    metrics_path,
                    provenance_path,
                    changed,
                )
            )

    def test_summary_identity_rejects_wrong_manifest_run(self):
        run = {
            "scene": "assets/cornell_box/cornell.scene",
            "eye": "1,2,3",
            "lookat": "0,0,0",
            "up": "0,1,0",
            "fov": 35,
            "experiment_seed": 11,
        }
        manifest = {
            "schema_version": 3,
            "pytorch_steps": 20,
            "torch_device": "cuda",
            "provenance_id": "batch-a",
            "cuda_learning_rate": 1.0,
        }
        summary = {
            "schema_version": 4,
            "scene": str(Path(run["scene"]).resolve()),
            "eye": run["eye"],
            "lookat": run["lookat"],
            "up": run["up"],
            "fov": run["fov"],
            "experiment_seed": run["experiment_seed"],
            "provenance_id": "batch-a",
            "cuda_learning_rate": 1.0,
            "reference": {
                "device": "cuda",
                "experiment_seed": 11,
                "pytorch": {"steps": 20},
            },
        }
        self.module.validate_summary_identity(
            run, summary, manifest, Path("valid-run")
        )
        summary["experiment_seed"] = 29
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.module.validate_summary_identity(
                run, summary, manifest, Path("wrong-run")
            )

    def test_objective_match_rejects_stale_artifact(self):
        self.assertAlmostEqual(
            self.module.require_matching_objective(
                "candidate", 10.00001, 10.0, Path("run")
            ),
            0.00001,
        )
        with self.assertRaisesRegex(AssertionError, "candidate objective"):
            self.module.require_matching_objective(
                "candidate", 11.0, 10.0, Path("run")
            )


if __name__ == "__main__":
    unittest.main()
