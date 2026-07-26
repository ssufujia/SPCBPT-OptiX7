"""Sweep the production CUDA mirror optimizer's initial learning rate."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_optimal_e_optimizers as comparison
import optimal_e_reference as reference


def rate_key(rate: float) -> str:
    return f"{rate:.9g}"


def parse_learning_rates(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            raise ValueError("learning rates must be comma-separated numbers")
        value = float(item)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("learning rates must be finite and positive")
        values.append(value)
    if len({rate_key(value) for value in values}) != len(values):
        raise ValueError("learning rates must be unique")
    return values


def build_validator_command(
    executable: Path,
    snapshot: Path,
    result: Path,
    learning_rate: float,
) -> list[str]:
    return [
        str(executable),
        f"--snapshot={snapshot}",
        f"--result={result}",
        f"--learning-rate={rate_key(learning_rate)}",
    ]


def relative_improvement(initial: float, final: float) -> float:
    return comparison.relative_improvement(initial, final)


def aggregate_rows(
    rows: Iterable[Mapping[str, object]],
    learning_rates: Iterable[float],
) -> dict[str, object]:
    materialized = list(rows)
    keys = [rate_key(rate) for rate in learning_rates]
    values = {
        key: {"improvements": [], "accepted_steps": [], "win_count": 0}
        for key in keys
    }
    for row in materialized:
        initial = float(row["initial_loss"])
        finals = {
            key: float(row["rates"][key]["final_loss"]) for key in keys
        }
        best = min(finals.values())
        for key, final in finals.items():
            values[key]["improvements"].append(
                relative_improvement(initial, final)
            )
            values[key]["accepted_steps"].append(
                int(row["rates"][key]["accepted_steps"])
            )
            if math.isclose(final, best, rel_tol=2e-5, abs_tol=1e-4):
                values[key]["win_count"] += 1

    rates = {}
    for key, rate_values in values.items():
        improvements = rate_values["improvements"]
        accepted_steps = rate_values["accepted_steps"]
        rates[key] = {
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
            "mean_accepted_steps": (
                statistics.fmean(accepted_steps) if accepted_steps else None
            ),
            "median_accepted_steps": (
                statistics.median(accepted_steps) if accepted_steps else None
            ),
            "win_count": rate_values["win_count"],
        }
    return {"run_count": len(materialized), "rates": rates}


def run_validator(
    executable: Path,
    snapshot: Path,
    result: Path,
    learning_rate: float,
) -> float:
    command = build_validator_command(
        executable, snapshot, result, learning_rate
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"CUDA validator failed for lr={learning_rate}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return elapsed


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = [
        "index",
        "scene",
        "camera",
        "seed",
        "snapshot_sha256",
        "learning_rate",
        "initial_loss",
        "final_loss",
        "relative_improvement",
        "accepted_steps",
        "cuda_objective_abs_error",
        "wall_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for key, result in row["rates"].items():
                writer.writerow(
                    {
                        "index": row["index"],
                        "scene": row["scene"],
                        "camera": row["camera"],
                        "seed": row["seed"],
                        "snapshot_sha256": row["snapshot_sha256"],
                        "learning_rate": key,
                        "initial_loss": row["initial_loss"],
                        "final_loss": result["final_loss"],
                        "relative_improvement": result[
                            "relative_improvement"
                        ],
                        "accepted_steps": result["accepted_steps"],
                        "cuda_objective_abs_error": result[
                            "cuda_objective_abs_error"
                        ],
                        "wall_seconds": result["wall_seconds"],
                    }
                )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--learning-rates",
        default="0.01,0.05,0.1,0.5,1.0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="sweep only the first N runs; 0 means all",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        learning_rates = parse_learning_rates(args.learning_rates)
        if args.count < 0:
            raise ValueError("--count cannot be negative")
        experiment_root = args.experiment_root.resolve()
        manifest_path = experiment_root / "manifest.json"
        manifest = comparison.read_json(manifest_path)
        runs = manifest["runs"]
        if int(manifest["run_count"]) != len(runs):
            raise RuntimeError("manifest run_count is inconsistent")
        selected_runs = runs[: args.count] if args.count else runs
        validator = (
            args.build_dir.resolve()
            / "bin"
            / "spcbpt_optimal_e_optimizer_test.exe"
        )
        if not validator.is_file():
            raise RuntimeError(f"missing CUDA validator: {validator}")
        device = str(reference.resolve_device(args.device))
        rows = []
        with tempfile.TemporaryDirectory(
            prefix="spcbpt-mirror-lr-"
        ) as temporary_directory:
            result_path = Path(temporary_directory) / "result.spcor"
            for position, run in enumerate(selected_runs, start=1):
                run_directory = experiment_root / comparison.run_name(run)
                snapshot_path = run_directory / "problem.spcoe"
                if not snapshot_path.is_file():
                    raise RuntimeError(f"missing snapshot: {snapshot_path}")
                problem = reference.load_export(snapshot_path, device)
                initial_loss, _ = comparison.evaluate_q(
                    problem,
                    problem["base_q"],
                    f"{snapshot_path} base_q",
                    device,
                )
                rate_results = {}
                for learning_rate in learning_rates:
                    elapsed = run_validator(
                        validator,
                        snapshot_path,
                        result_path,
                        learning_rate,
                    )
                    cuda_result = reference.load_cuda_result(
                        result_path, device
                    )
                    final_loss, row_sum_error = comparison.evaluate_q(
                        problem,
                        cuda_result["final_q"],
                        f"{snapshot_path} lr={learning_rate}",
                        device,
                    )
                    absolute_error = abs(
                        float(cuda_result["final_objective"]) - final_loss
                    )
                    tolerance = 1e-4 + 2e-5 * abs(final_loss)
                    if absolute_error > tolerance:
                        raise AssertionError(
                            f"objective mismatch for {snapshot_path}, "
                            f"lr={learning_rate}: {absolute_error} > {tolerance}"
                        )
                    key = rate_key(learning_rate)
                    rate_results[key] = {
                        "final_loss": final_loss,
                        "relative_improvement": relative_improvement(
                            initial_loss, final_loss
                        ),
                        "accepted_steps": int(
                            cuda_result["accepted_steps"]
                        ),
                        "cuda_final_objective": float(
                            cuda_result["final_objective"]
                        ),
                        "cuda_objective_abs_error": absolute_error,
                        "max_row_sum_error": row_sum_error,
                        "wall_seconds": elapsed,
                    }
                rows.append(
                    {
                        "index": int(run["index"]),
                        "scene": str(run["scene_id"]),
                        "camera": str(run["camera"]),
                        "camera_index": int(run["camera_index"]),
                        "seed": int(run["experiment_seed"]),
                        "snapshot_sha256": comparison.file_sha256(
                            snapshot_path
                        ),
                        "initial_loss": initial_loss,
                        "rates": rate_results,
                    }
                )
                print(
                    f"SWEEP {position}/{len(selected_runs)} "
                    f"{comparison.run_name(run)}"
                )

        aggregate = aggregate_rows(rows, learning_rates)
        scene_aggregates = {
            scene: aggregate_rows(
                (row for row in rows if row["scene"] == scene),
                learning_rates,
            )
            for scene in sorted({row["scene"] for row in rows})
        }
        suffix = f"-first{args.count}" if args.count else ""
        json_path = (
            experiment_root
            / f"cuda-mirror-learning-rate-sweep{suffix}.json"
        )
        csv_path = (
            experiment_root
            / f"cuda-mirror-learning-rate-sweep{suffix}.csv"
        )
        comparison.write_json(
            json_path,
            {
                "schema_version": 2,
                "manifest_sha256": comparison.file_sha256(manifest_path),
                "validator_sha256": comparison.file_sha256(validator),
                "provenance": {
                    "manifest": {
                        "path": str(manifest_path),
                        "sha256": comparison.file_sha256(manifest_path),
                    },
                    "validator": {
                        "path": str(validator),
                        "sha256": comparison.file_sha256(validator),
                    },
                    "sweep_script": {
                        "path": str(Path(__file__).resolve()),
                        "sha256": comparison.file_sha256(
                            Path(__file__).resolve()
                        ),
                    },
                    "reference_script": {
                        "path": str(Path(reference.__file__).resolve()),
                        "sha256": comparison.file_sha256(
                            Path(reference.__file__).resolve()
                        ),
                    },
                    "comparison_script": {
                        "path": str(Path(comparison.__file__).resolve()),
                        "sha256": comparison.file_sha256(
                            Path(comparison.__file__).resolve()
                        ),
                    },
                },
                "device": device,
                "learning_rates": learning_rates,
                "completed": len(rows),
                "expected": len(selected_runs),
                "aggregate": aggregate,
                "scene_aggregates": scene_aggregates,
                "rows": rows,
            },
        )
        write_csv(csv_path, rows)
        print(
            f"CUDA_MIRROR_LEARNING_RATE_SWEEP_OK: "
            f"{len(rows)}/{len(selected_runs)}"
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
        print(
            f"sweep_cuda_mirror_learning_rate: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
