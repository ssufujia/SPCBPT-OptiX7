import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from scripts import optimal_e_reference as ref


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OptimalEReferenceTests(unittest.TestCase):
    def test_export_format_describes_active_light_normalization(self):
        self.assertIn("active_light_count", ref.EXPORT_FORMAT)
        self.assertIn("inactive columns are exactly zero", ref.EXPORT_FORMAT)
        self.assertIn("active columns", ref.EXPORT_FORMAT)

    def test_autograd_matches_finite_difference(self):
        result = ref.check_autograd(seed=7)
        self.assertLess(result["max_abs_error"], 2e-6)

    def test_probability_leaf_gradient_matches_sparse_oracle(self):
        result = ref.check_probability_gradient()
        self.assertGreater(result["min_q"], 0.0)
        self.assertLess(result["max_row_sum_error"], 1e-15)
        self.assertLess(result["max_abs_error"], 1e-12)

    def test_objective_is_convex_in_probabilities(self):
        result = ref.check_convexity(seed=11)
        self.assertLessEqual(result["jensen_gap"], 1e-10)

    def test_optimizer_decreases_loss(self):
        result = ref.check_loss_descent(seed=13, steps=150)
        self.assertLess(result["final_loss"], result["initial_loss"] * 0.95)

    def test_optimizer_returns_base_distribution(self):
        problem = ref.make_synthetic_problem()
        base_q, _ = ref.optimize(
            ref._sample_tensors(problem),
            problem["num_eye"],
            problem["num_light"],
            problem["conservative_rate"],
            steps=2,
        )
        self.assertTrue(torch.all(base_q >= 0.0))
        self.assertTrue(
            torch.allclose(
                base_q.sum(dim=1),
                torch.ones(
                    problem["num_eye"],
                    dtype=base_q.dtype,
                    device=base_q.device,
                ),
            )
        )
        effective_e = ref.mix_conservative_probabilities(
            base_q, problem["conservative_rate"]
        )
        self.assertFalse(torch.equal(base_q, effective_e))

    def test_inactive_light_columns_are_conditioned_out(self):
        base_q = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64
        )
        active = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        effective_e = ref.mix_conservative_probabilities(
            base_q, 0.2, active
        )
        self.assertEqual(effective_e[0, 1].item(), 0.0)
        self.assertEqual(effective_e[0, 3].item(), 0.0)
        self.assertAlmostEqual(effective_e.sum().item(), 1.0)
        self.assertAlmostEqual(effective_e[0, 0].item(), 0.3)
        self.assertAlmostEqual(effective_e[0, 2].item(), 0.7)

    def test_inactive_light_samples_must_have_zero_contribution(self):
        problem = ref.make_synthetic_problem()
        samples = ref._sample_tensors(problem)
        device = samples["f2"].device
        inactive_light = int(samples["light_index"][0].item())
        active = torch.ones(
            problem["num_light"], dtype=torch.bool, device=device
        )
        active[inactive_light] = False
        inactive_samples = samples["light_index"] == inactive_light
        samples["peak_pdf"][inactive_samples] = 0.0
        logits = torch.zeros(
            (problem["num_eye"], problem["num_light"]),
            dtype=torch.float64,
            device=device,
        )

        loss = ref.reference_loss(
            logits,
            samples,
            problem["conservative_rate"],
            active,
        )
        self.assertTrue(torch.isfinite(loss))

        samples["peak_pdf"][inactive_samples.nonzero()[0]] = 1.0
        with self.assertRaisesRegex(ValueError, "inactive light"):
            ref.reference_loss(
                logits,
                samples,
                problem["conservative_rate"],
                active,
            )

    def test_invalid_inputs_are_rejected(self):
        import tempfile

        import numpy as np

        problem = ref.make_synthetic_problem()
        samples = ref._sample_tensors(problem)
        e = torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64)

        with self.subTest(case="zero num_light"):
            with self.assertRaisesRegex(ValueError, "num_light"):
                ref.effective_probabilities(torch.empty((2, 0)), 0.2)

        invalid_cases = [("E", None, float("nan"))]
        for name in ("f2", "p0", "peak_pdf"):
            invalid_cases.extend(((name, name, float("nan")), (f"negative {name}", name, -1.0)))
        for case, field, value in invalid_cases:
            with self.subTest(case=case):
                bad_e = e.clone()
                bad_samples = {name: tensor.clone() for name, tensor in samples.items()}
                if field is None:
                    bad_e[0, 0] = value
                else:
                    bad_samples[field][0] = value
                with self.assertRaises(ValueError):
                    ref.loss_from_probabilities(bad_e, bad_samples)

        with self.subTest(case="missing export field"):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "incomplete.npz"
                np.savez(path, schema_version=np.array(1, dtype=np.int32))
                with self.assertRaisesRegex(ValueError, "missing export arrays"):
                    ref.load_export(path, "cpu")

    def test_binary_snapshot_matches_cuda_result(self):
        import json
        import struct
        import tempfile

        import numpy as np

        f2 = np.array([4.0, 2.0], dtype="<f4")
        p0 = np.array([0.5, 0.75], dtype="<f4")
        offsets = np.array([0, 2, 3], dtype="<i4")
        matrix_indices = np.array([0, 1, 4], dtype="<i4")
        peak_pdf = np.array([2.0, 1.0, 3.0], dtype="<f4")
        active = np.array([1, 1, 0], dtype="<i4")
        base_q = np.array(
            [[0.75, 0.25, 0.0], [0.5, 0.5, 0.0]], dtype="<f4"
        )
        conservative_rate = 0.2

        effective = (1.0 - conservative_rate) * base_q
        effective[:, :2] += conservative_rate / 2
        density = np.array(
            [
                p0[0] + 2.0 * effective[0, 0] + effective[0, 1],
                p0[1] + 3.0 * effective[1, 1],
            ],
            dtype=np.float64,
        )
        objective = float(np.sum(f2.astype(np.float64) / density))
        gradient = np.zeros(6, dtype="<f4")
        gradient[0] = -(1.0 - conservative_rate) * f2[0] * 2.0 / density[0] ** 2
        gradient[1] = -(1.0 - conservative_rate) * f2[0] / density[0] ** 2
        gradient[4] = -(1.0 - conservative_rate) * f2[1] * 3.0 / density[1] ** 2

        with tempfile.TemporaryDirectory() as directory:
            snapshot_path = Path(directory) / "problem.spcoe"
            result_path = Path(directory) / "cuda.spcor"
            snapshot_header = struct.pack(
                "<8sIIQQIIIf",
                b"SPCBE001",
                1,
                0x01020304,
                len(f2),
                len(peak_pdf),
                2,
                3,
                2,
                conservative_rate,
            )
            snapshot_path.write_bytes(
                snapshot_header
                + f2.tobytes()
                + p0.tobytes()
                + peak_pdf.tobytes()
                + offsets.tobytes()
                + matrix_indices.tobytes()
                + active.tobytes()
                + base_q.tobytes()
            )
            result_header = struct.pack(
                "<8sIIIIIff",
                b"SPCBR001",
                1,
                0x01020304,
                2,
                3,
                0,
                objective,
                objective,
            )
            result_path.write_bytes(
                result_header
                + gradient.tobytes()
                + base_q.tobytes()
            )

            problem = ref.load_export(snapshot_path, "cpu")
            self.assertEqual(problem["experiment_seed"], 0)
            cuda_result = ref.load_cuda_result(result_path, "cpu")
            metrics = ref.compare_cuda_result(problem, cuda_result)
            self.assertLess(metrics["gradient_max_abs_error"], 1e-5)
            self.assertLess(metrics["initial_objective_abs_error"], 1e-5)
            self.assertLess(metrics["final_objective_abs_error"], 1e-5)

            cuda_result["initial_gradient"][0] += 1.0
            with self.assertRaisesRegex(AssertionError, "gradient"):
                ref.compare_cuda_result(problem, cuda_result)

            snapshot_header = struct.pack(
                "<8sIIQQIIIIf",
                b"SPCBE001",
                2,
                0x01020304,
                len(f2),
                len(peak_pdf),
                2,
                3,
                2,
                47,
                conservative_rate,
            )
            snapshot_path.write_bytes(
                snapshot_header
                + f2.tobytes()
                + p0.tobytes()
                + peak_pdf.tobytes()
                + offsets.tobytes()
                + matrix_indices.tobytes()
                + active.tobytes()
                + base_q.tobytes()
            )
            problem = ref.load_export(snapshot_path, "cpu")
            self.assertEqual(problem["experiment_seed"], 47)

            metrics_path = Path(directory) / "metrics.json"
            output_path = Path(directory) / "pytorch_base_q.npy"
            raw_output_path = Path(directory) / "pytorch_base_q.f32"
            exit_code = ref.main(
                [
                    "--input",
                    str(snapshot_path),
                    "--cuda-result",
                    str(result_path),
                    "--output",
                    str(output_path),
                    "--raw-output",
                    str(raw_output_path),
                    "--metrics-output",
                    str(metrics_path),
                    "--steps",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(exit_code, 0)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["device"], "cpu")
            self.assertEqual(metrics["pytorch"]["steps"], 1)
            self.assertIn("final_loss", metrics["pytorch"])
            self.assertIn("cross_check", metrics)
            raw_q = np.fromfile(raw_output_path, dtype="<f4")
            self.assertEqual(raw_q.size, 6)
            self.assertTrue(np.all(np.isfinite(raw_q)))
            np.testing.assert_array_equal(
                raw_q,
                np.load(output_path).astype("<f4").reshape(-1),
            )


if __name__ == "__main__":
    unittest.main()
