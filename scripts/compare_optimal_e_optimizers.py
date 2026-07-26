"""Compare production, PyTorch, and fixed legacy CUDA optimizers.

The input is a completed directory produced by
``scripts/run_optimal_e_multiscene.ps1``. Generated candidate matrices and
result tables stay under that ignored experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import optimal_e_reference as reference


METHOD_FIELDS = {
    "cuda_mirror": "cuda_mirror_final_loss",
    "pytorch_adam": "pytorch_adam_final_loss",
    "legacy_adam_original": "legacy_adam_original_final_loss",
    "legacy_adam_matched": "legacy_adam_matched_final_loss",
}


def run_name(run: Mapping[str, object]) -> str:
    return (
        f"{int(run['index']):03d}-{run['scene_id']}"
        f"-c{int(run['camera_index']):02d}-{run['camera']}"
        f"-s{int(run['experiment_seed'])}"
    )


def relative_improvement(initial: float, final: float) -> float:
    if not math.isfinite(initial) or initial <= 0.0:
        raise ValueError("initial loss must be finite and positive")
    if not math.isfinite(final):
        raise ValueError("final loss must be finite")
    return (initial - final) / initial


def require_matching_objective(
    label: str,
    actual: float,
    expected: float,
    directory: Path,
) -> float:
    absolute_error = abs(actual - expected)
    tolerance = 1e-4 + 2e-5 * abs(expected)
    if absolute_error > tolerance:
        raise AssertionError(
            f"{label} objective mismatch in {directory}: "
            f"abs_error={absolute_error}, tolerance={tolerance}"
        )
    return absolute_error


def aggregate_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    materialized = list(rows)
    methods = {
        name: {
            "relative_improvements": [],
            "win_count": 0,
        }
        for name in METHOD_FIELDS
    }
    for row in materialized:
        initial = float(row["initial_loss"])
        finals = {
            name: float(row[field])
            for name, field in METHOD_FIELDS.items()
        }
        best = min(finals.values())
        for name, final in finals.items():
            methods[name]["relative_improvements"].append(
                relative_improvement(initial, final)
            )
            if math.isclose(final, best, rel_tol=2e-5, abs_tol=1e-4):
                methods[name]["win_count"] += 1

    result_methods = {}
    for name, values in methods.items():
        improvements = values["relative_improvements"]
        result_methods[name] = {
            "mean_relative_improvement": (
                statistics.fmean(improvements) if improvements else None
            ),
            "median_relative_improvement": (
                statistics.median(improvements) if improvements else None
            ),
            "min_relative_improvement": (
                min(improvements) if improvements else None
            ),
            "max_relative_improvement": (
                max(improvements) if improvements else None
            ),
            "win_count": values["win_count"],
        }
    return {
        "run_count": len(materialized),
        "methods": result_methods,
    }


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON: {path}") from exc


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_summary_identity(
    run: Mapping[str, object],
    summary: Mapping[str, object],
    manifest: Mapping[str, object],
    directory: Path,
) -> None:
    manifest_schema = int(manifest["schema_version"])
    expected_summary_schema = 4 if manifest_schema >= 3 else 3
    reference_metrics = summary.get("reference")
    if not isinstance(reference_metrics, Mapping):
        raise RuntimeError(f"summary is missing reference metrics: {directory}")
    pytorch_metrics = reference_metrics.get("pytorch")
    if not isinstance(pytorch_metrics, Mapping):
        raise RuntimeError(f"summary is missing PyTorch metrics: {directory}")

    expected_scene = Path(str(run["scene"])).resolve()
    actual_scene = Path(str(summary.get("scene", ""))).resolve()
    identity_matches = (
        int(summary.get("schema_version", -1)) == expected_summary_schema
        and actual_scene == expected_scene
        and summary.get("eye") == run["eye"]
        and summary.get("lookat") == run["lookat"]
        and summary.get("up") == run["up"]
        and math.isclose(
            float(summary.get("fov", math.nan)),
            float(run["fov"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and int(summary.get("experiment_seed", -1))
        == int(run["experiment_seed"])
        and int(reference_metrics.get("experiment_seed", -1))
        == int(run["experiment_seed"])
        and int(pytorch_metrics.get("steps", -1))
        == int(manifest["pytorch_steps"])
    )
    expected_device = str(manifest["torch_device"])
    if expected_device != "auto":
        identity_matches = (
            identity_matches
            and reference_metrics.get("device") == expected_device
        )
    if manifest_schema >= 3:
        identity_matches = (
            identity_matches
            and summary.get("provenance_id") == manifest.get("provenance_id")
            and math.isclose(
                float(summary.get("cuda_learning_rate", math.nan)),
                float(manifest["cuda_learning_rate"]),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        )
    if not identity_matches:
        raise RuntimeError(f"summary identity mismatch: {directory}")


def evaluate_q(
    problem: Mapping[str, object],
    q,
    label: str,
    device: str,
) -> tuple[float, float]:
    num_eye = int(problem["num_eye"])
    num_light = int(problem["num_light"])
    if tuple(q.shape) != (num_eye, num_light):
        raise ValueError(
            f"{label} has shape {tuple(q.shape)}, "
            f"expected {(num_eye, num_light)}"
        )
    q = q.to(dtype=reference.torch.float64, device=device)
    active = problem.get("active_light")
    if active is None:
        active = reference.torch.ones(
            num_light, dtype=reference.torch.bool, device=device
        )
    active_float = active.to(dtype=q.dtype)
    if not reference.torch.all(reference.torch.isfinite(q)).item():
        raise ValueError(f"{label} contains non-finite values")
    if reference.torch.any(q < -1e-7).item():
        raise ValueError(f"{label} contains negative probabilities")
    inactive_error = (
        reference.torch.abs(q * (~active).to(dtype=q.dtype)).max().item()
    )
    row_sum_error = (
        reference.torch.abs(
            (q * active_float).sum(dim=1)
            - reference.torch.ones(num_eye, dtype=q.dtype, device=device)
        )
        .max()
        .item()
    )
    if inactive_error > 1e-6 or row_sum_error > 2e-5:
        raise ValueError(
            f"{label} is infeasible: inactive_error={inactive_error}, "
            f"row_sum_error={row_sum_error}"
        )

    uniform = active_float / active_float.sum()
    effective = (
        (1.0 - float(problem["conservative_rate"])) * q
        + float(problem["conservative_rate"]) * uniform
    ) * active_float
    loss = reference.loss_from_probabilities(
        effective, reference._sample_tensors(problem)
    )
    return float(loss.item()), float(row_sum_error)


def evaluate_raw_q(
    problem: Mapping[str, object],
    q_path: Path,
    device: str,
) -> tuple[float, float]:
    num_eye = int(problem["num_eye"])
    num_light = int(problem["num_light"])
    raw_q = np.fromfile(q_path, dtype="<f4")
    expected_size = num_eye * num_light
    if raw_q.size != expected_size:
        raise ValueError(
            f"{q_path} has {raw_q.size} values, expected {expected_size}"
        )

    q = reference.torch.as_tensor(
        raw_q.reshape(num_eye, num_light),
        device=device,
    )
    return evaluate_q(problem, q, str(q_path), device)


def load_cached_legacy_variant(
    q_path: Path,
    metrics_path: Path,
    provenance_path: Path,
    expected_provenance: Mapping[str, object],
) -> dict[str, object] | None:
    if not (
        q_path.is_file()
        and metrics_path.is_file()
        and provenance_path.is_file()
    ):
        return None
    metrics = read_json(metrics_path)
    provenance = read_json(provenance_path)
    cached_identity = {
        key: provenance.get(key) for key in expected_provenance
    }
    metrics_match = (
        metrics.get("schema_version") == 1
        and metrics.get("optimizer")
        == "fixed_legacy_sigmoid_adam_cuda"
        and metrics.get("steps") == expected_provenance["steps"]
        and math.isclose(
            float(metrics.get("learning_rate", math.nan)),
            float(expected_provenance["learning_rate"]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )
    q_matches = provenance.get("q_sha256") == file_sha256(q_path)
    if (
        cached_identity == dict(expected_provenance)
        and metrics_match
        and q_matches
    ):
        return metrics
    return None


def run_legacy_variant(
    executable: Path,
    executable_sha256: str,
    snapshot_path: Path,
    snapshot_sha256: str,
    run_dir: Path,
    tag: str,
    steps: int,
    learning_rate: float,
    force: bool,
) -> tuple[dict[str, object], Path, float]:
    q_path = run_dir / f"{tag}.f32"
    metrics_path = run_dir / f"{tag}.json"
    provenance_path = run_dir / f"{tag}.provenance.json"
    log_path = run_dir / f"{tag}.log"
    expected_provenance = {
        "schema_version": 1,
        "optimizer": "fixed_legacy_sigmoid_adam_cuda",
        "steps": steps,
        "learning_rate": learning_rate,
        "snapshot_sha256": snapshot_sha256,
        "executable_sha256": executable_sha256,
    }
    if not force:
        cached_metrics = load_cached_legacy_variant(
            q_path,
            metrics_path,
            provenance_path,
            expected_provenance,
        )
        if cached_metrics is not None:
            return cached_metrics, q_path, 0.0

    command = [
        str(executable),
        f"--snapshot={snapshot_path}",
        f"--output-q={q_path}",
        f"--metrics-output={metrics_path}",
        f"--steps={steps}",
        f"--learning-rate={learning_rate:.9g}",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall_seconds = time.perf_counter() - started
    log_path.write_text(
        "COMMAND: "
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"legacy optimizer failed ({completed.returncode}); see {log_path}"
        )
    metrics = read_json(metrics_path)
    if (
        metrics.get("schema_version") != 1
        or metrics.get("optimizer")
        != "fixed_legacy_sigmoid_adam_cuda"
        or metrics.get("steps") != steps
        or not math.isclose(
            float(metrics.get("learning_rate", math.nan)),
            learning_rate,
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError(f"legacy metrics identity mismatch: {metrics_path}")
    provenance = dict(expected_provenance)
    provenance["q_sha256"] = file_sha256(q_path)
    write_json(provenance_path, provenance)
    return metrics, q_path, wall_seconds


def compare_run(
    run: Mapping[str, object],
    manifest: Mapping[str, object],
    experiment_root: Path,
    legacy_executable: Path,
    legacy_executable_sha256: str,
    device: str,
    steps: int,
    original_learning_rate: float,
    matched_learning_rate: float,
    force: bool,
) -> dict[str, object]:
    directory = experiment_root / run_name(run)
    snapshot_path = directory / "problem.spcoe"
    summary_path = directory / "summary.json"
    if not snapshot_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"incomplete multiscene run: {directory}")
    summary = read_json(summary_path)
    validate_summary_identity(run, summary, manifest, directory)
    problem = reference.load_export(snapshot_path, device)
    snapshot_sha256 = file_sha256(snapshot_path)

    variants = (
        ("legacy_adam_original", original_learning_rate),
        ("legacy_adam_matched", matched_learning_rate),
    )
    legacy_results = {}
    for tag, learning_rate in variants:
        metrics, q_path, wall_seconds = run_legacy_variant(
            legacy_executable,
            legacy_executable_sha256,
            snapshot_path,
            snapshot_sha256,
            directory,
            tag,
            steps,
            learning_rate,
            force,
        )
        pytorch_loss, row_sum_error = evaluate_raw_q(
            problem, q_path, device
        )
        cuda_loss = float(metrics["final_loss"])
        absolute_error = require_matching_objective(
            tag, cuda_loss, pytorch_loss, directory
        )
        legacy_results[tag] = {
            "metrics": metrics,
            "pytorch_final_loss": pytorch_loss,
            "objective_abs_error": absolute_error,
            "max_row_sum_error": row_sum_error,
            "wall_seconds": wall_seconds,
        }

    cross_check = summary["reference"]["cross_check"]
    pytorch = summary["reference"]["pytorch"]
    cuda_result_path = directory / "cuda.spcor"
    pytorch_q_path = directory / "pytorch_base_q.f32"
    if not cuda_result_path.is_file() or not pytorch_q_path.is_file():
        raise RuntimeError(f"missing optimizer q artifact: {directory}")
    cuda_result = reference.load_cuda_result(cuda_result_path, device)
    cuda_mirror_loss, cuda_row_sum_error = evaluate_q(
        problem,
        cuda_result["final_q"],
        str(cuda_result_path),
        device,
    )
    pytorch_loss, pytorch_row_sum_error = evaluate_raw_q(
        problem, pytorch_q_path, device
    )
    cuda_initial = float(cross_check["cuda_initial_objective"])
    pytorch_initial = float(pytorch["initial_loss"])
    initial_tolerance = 1e-4 + 2e-5 * abs(pytorch_initial)
    if abs(cuda_initial - pytorch_initial) > initial_tolerance:
        raise AssertionError(f"initial loss mismatch in {directory}")
    cuda_summary_error = require_matching_objective(
        "CUDA mirror artifact",
        cuda_mirror_loss,
        float(cross_check["cuda_final_objective"]),
        directory,
    )
    pytorch_summary_error = require_matching_objective(
        "PyTorch artifact",
        pytorch_loss,
        float(pytorch["final_loss"]),
        directory,
    )

    return {
        "index": int(run["index"]),
        "scene": str(run["scene_id"]),
        "camera": str(run["camera"]),
        "camera_index": int(run["camera_index"]),
        "seed": int(run["experiment_seed"]),
        "num_paths": int(problem["f2"].numel()),
        "num_nodes": int(problem["peak_pdf"].numel()),
        "active_light_count": int(
            problem["active_light"].sum().item()
            if problem.get("active_light") is not None
            else problem["num_light"]
        ),
        "snapshot_bytes": snapshot_path.stat().st_size,
        "initial_loss": pytorch_initial,
        "cuda_mirror_final_loss": cuda_mirror_loss,
        "pytorch_adam_final_loss": pytorch_loss,
        "legacy_adam_original_final_loss": float(
            legacy_results["legacy_adam_original"][
                "pytorch_final_loss"
            ]
        ),
        "legacy_adam_matched_final_loss": float(
            legacy_results["legacy_adam_matched"][
                "pytorch_final_loss"
            ]
        ),
        "cuda_mirror_production_final_loss": float(
            cross_check["cuda_final_objective"]
        ),
        "pytorch_adam_reference_final_loss": float(
            pytorch["final_loss"]
        ),
        "legacy_adam_original_production_final_loss": float(
            legacy_results["legacy_adam_original"]["metrics"]["final_loss"]
        ),
        "legacy_adam_matched_production_final_loss": float(
            legacy_results["legacy_adam_matched"]["metrics"]["final_loss"]
        ),
        "cuda_mirror_relative_improvement": relative_improvement(
            pytorch_initial, cuda_mirror_loss
        ),
        "pytorch_adam_relative_improvement": relative_improvement(
            pytorch_initial, pytorch_loss
        ),
        "legacy_adam_original_relative_improvement": relative_improvement(
            pytorch_initial,
            float(
                legacy_results["legacy_adam_original"][
                    "pytorch_final_loss"
                ]
            ),
        ),
        "legacy_adam_matched_relative_improvement": relative_improvement(
            pytorch_initial,
            float(
                legacy_results["legacy_adam_matched"][
                    "pytorch_final_loss"
                ]
            ),
        ),
        "legacy_adam_original_pytorch_objective_abs_error": float(
            legacy_results["legacy_adam_original"]["objective_abs_error"]
        ),
        "legacy_adam_matched_pytorch_objective_abs_error": float(
            legacy_results["legacy_adam_matched"]["objective_abs_error"]
        ),
        "current_cuda_pytorch_objective_abs_error": float(
            cuda_summary_error
        ),
        "pytorch_artifact_objective_abs_error": pytorch_summary_error,
        "pytorch_cuda_reverse_objective_abs_error": float(
            summary["reverse_cross_check"]["absolute_error"]
        ),
        "cuda_mirror_max_row_sum_error": cuda_row_sum_error,
        "pytorch_adam_max_row_sum_error": pytorch_row_sum_error,
        "cuda_mirror_seconds": float(
            summary["timings_seconds"]["cuda"]
        ),
        "pytorch_adam_seconds": float(
            summary["timings_seconds"]["pytorch"]
        ),
        "legacy_adam_original_internal_seconds": float(
            legacy_results["legacy_adam_original"]["metrics"][
                "elapsed_seconds"
            ]
        ),
        "legacy_adam_matched_internal_seconds": float(
            legacy_results["legacy_adam_matched"]["metrics"][
                "elapsed_seconds"
            ]
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("build/experiments/optimal-e-multiscene-v4"),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/release-optix9"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="compare only the first N manifest runs; 0 means all",
    )
    parser.add_argument(
        "--legacy-original-learning-rate", type=float, default=0.01
    )
    parser.add_argument(
        "--legacy-matched-learning-rate", type=float, default=0.05
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_root = args.experiment_root.resolve()
    legacy_executable = (
        args.build_dir.resolve()
        / "bin"
        / "spcbpt_legacy_optimal_e_optimizer_test.exe"
    )
    try:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        if (
            not math.isfinite(args.legacy_original_learning_rate)
            or args.legacy_original_learning_rate <= 0.0
            or not math.isfinite(args.legacy_matched_learning_rate)
            or args.legacy_matched_learning_rate <= 0.0
        ):
            raise ValueError("legacy learning rates must be finite and positive")
        if not legacy_executable.is_file():
            raise RuntimeError(
                f"missing legacy optimizer executable: {legacy_executable}"
            )
        manifest_path = experiment_root / "manifest.json"
        manifest = read_json(manifest_path)
        runs = manifest["runs"]
        if int(manifest["run_count"]) != len(runs):
            raise RuntimeError("manifest run_count is inconsistent")
        if args.count < 0:
            raise ValueError("--count cannot be negative")
        selected_runs = runs[: args.count] if args.count else runs
        device = str(reference.resolve_device(args.device))
        executable_sha256 = file_sha256(legacy_executable)
        comparison_config = {
            "schema_version": 1,
            "manifest_sha256": file_sha256(manifest_path),
            "legacy_executable_sha256": executable_sha256,
            "steps": args.steps,
            "legacy_original_learning_rate": (
                args.legacy_original_learning_rate
            ),
            "legacy_matched_learning_rate": (
                args.legacy_matched_learning_rate
            ),
            "evaluation_device": device,
        }
        write_json(
            experiment_root / "optimizer-comparison-config.json",
            comparison_config,
        )
        rows = []
        for position, run in enumerate(selected_runs, start=1):
            row = compare_run(
                run,
                manifest,
                experiment_root,
                legacy_executable,
                executable_sha256,
                device,
                args.steps,
                args.legacy_original_learning_rate,
                args.legacy_matched_learning_rate,
                args.force,
            )
            rows.append(row)
            print(
                f"COMPARE {position}/{len(runs)} "
                f"{run_name(run)}"
            )

        scene_aggregates = {
            scene: aggregate_rows(
                row for row in rows if row["scene"] == scene
            )
            for scene in sorted({str(row["scene"]) for row in rows})
        }
        report = {
            "schema_version": 1,
            "experiment_root": str(experiment_root),
            "steps": args.steps,
            "legacy_original_learning_rate": (
                args.legacy_original_learning_rate
            ),
            "legacy_matched_learning_rate": (
                args.legacy_matched_learning_rate
            ),
            "device": device,
            "config": comparison_config,
            "completed": len(rows),
            "expected": int(manifest["run_count"]),
            "data": {
                "total_paths": sum(
                    int(row["num_paths"]) for row in rows
                ),
                "total_nodes": sum(
                    int(row["num_nodes"]) for row in rows
                ),
                "total_snapshot_bytes": sum(
                    int(row["snapshot_bytes"]) for row in rows
                ),
            },
            "aggregate": aggregate_rows(rows),
            "scene_aggregates": scene_aggregates,
            "rows": rows,
        }
        write_json(experiment_root / "optimizer-comparison.json", report)
        write_csv(experiment_root / "optimizer-comparison.csv", rows)
        print(
            "OPTIMIZER_COMPARISON_OK: "
            f"{len(rows)}/{manifest['run_count']}"
        )
        return 0
    except (
        AssertionError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"compare_optimal_e_optimizers: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
