"""PyTorch reference for the SPCBPT optimal-E objective.

The renderer's real-sample export is one versioned ``.spcoe`` binary file.
Binary schema 1 has no experiment seed and is read as ``experiment_seed=0``.
Binary schema 2 adds a little-endian uint32 ``experiment_seed`` immediately
before ``conservative_rate``; both schemas store ``active_light_count`` in
the header. ``.npz`` schema 1 remains supported for small hand-authored
fixtures:

    schema_version       int32 scalar, 1 for .npz fixtures
    num_eye              int32 scalar, number of probability rows
    num_light            int32 scalar, number of entries per row
    conservative_rate    float64 scalar, normally 0.2
    f2                    float32/float64 [N]
    p0                    float32/float64 [N]
    path_offsets          int64 [N + 1], starts at 0 and ends at M
    eye_index             int32/int64 [M]
    light_index           int32/int64 [M]
    peak_pdf              float32/float64 [M]
    active_light          int32 [num_light]
    base_q                float32 [num_eye, num_light]

For node k in path i, the implicit design matrix contributes
``peak_pdf[k] * E[eye_index[k], light_index[k]]``. Therefore:

    loss(E) = sum_i f2[i] / (
        p0[i] + sum_{k=path_offsets[i]:path_offsets[i+1]}
            peak_pdf[k] * E[eye_index[k], light_index[k]]
    )

Let ``A`` be the active-light set and
``active_light_count = len(A)``. Each row first uses a masked softmax:

    q[row, j] = exp(logits[row, j]) / sum_{k in A} exp(logits[row, k])
        for j in A, and 0 otherwise

The effective probabilities are then:

    E[row, j] = (1-rate) * q[row, j] + rate / active_light_count
        for active columns, while inactive columns are exactly zero

When an existing ``base_q`` is supplied instead of logits, only its active
columns are normalized to sum to one before applying the same conservative
mixture.

Run ``python scripts/optimal_e_reference.py`` for the synthetic self-test, or
pass ``--input problem.spcoe --cuda-result cuda.spcor`` for the real-data
PyTorch/CUDA cross-check.
By default the script picks ``cuda`` when available, otherwise ``cpu``
(``--device auto|cpu|cuda|cuda:N``).
PyTorch is optional for the renderer and is never installed by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


EXPORT_FORMAT = __doc__.split("Run ``python", 1)[0].strip()


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the offline optimal-E reference. "
            "Install a matching wheel from https://pytorch.org/get-started/locally/; "
            "this script does not install dependencies."
        ) from TORCH_IMPORT_ERROR


def default_device() -> str:
    """Prefer CUDA when the installed torch build can use it."""
    require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: str | None = None) -> torch.device:
    require_torch()
    chosen = default_device() if device in (None, "", "auto") else device
    resolved = torch.device(chosen)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "requested CUDA device but torch.cuda.is_available() is False"
        )
    return resolved


def mix_conservative_probabilities(
    base_q: torch.Tensor,
    conservative_rate: float,
    active_light: torch.Tensor | None = None,
) -> torch.Tensor:
    require_torch()
    if base_q.ndim != 2:
        raise ValueError("base_q must have shape [num_eye, num_light]")
    if base_q.shape[1] == 0:
        raise ValueError("num_light must be positive")
    if not 0.0 <= conservative_rate < 1.0:
        raise ValueError("conservative_rate must be in [0, 1)")
    if active_light is None:
        active = torch.ones(
            base_q.shape[1], dtype=torch.bool, device=base_q.device
        )
    else:
        active = torch.as_tensor(active_light, device=base_q.device)
        if active.ndim != 1 or active.shape[0] != base_q.shape[1]:
            raise ValueError("active_light must have shape [num_light]")
        if not torch.all((active == 0) | (active == 1)).item():
            raise ValueError("active_light must contain only zero or one")
        active = active.to(dtype=torch.bool)
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError("active_light must contain at least one active column")
    active_float = active.to(dtype=base_q.dtype)
    masked_q = base_q * active_float
    row_sum = masked_q.sum(dim=1, keepdim=True)
    uniform = active_float / active_count
    normalized_q = torch.where(
        row_sum > 0.0,
        masked_q / torch.clamp_min(row_sum, torch.finfo(base_q.dtype).tiny),
        uniform.expand_as(base_q),
    )
    return (
        (1.0 - conservative_rate) * normalized_q
        + conservative_rate * uniform
    )


def effective_probabilities(
    logits: torch.Tensor,
    conservative_rate: float,
    active_light: torch.Tensor | None = None,
) -> torch.Tensor:
    require_torch()
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_eye, num_light]")
    if active_light is None:
        base_q = torch.softmax(logits, dim=1)
    else:
        active = torch.as_tensor(
            active_light, dtype=torch.bool, device=logits.device
        )
        if active.ndim != 1 or active.shape[0] != logits.shape[1]:
            raise ValueError("active_light must have shape [num_light]")
        if not torch.any(active).item():
            raise ValueError("active_light must contain at least one active column")
        base_q = torch.softmax(logits.masked_fill(~active, -torch.inf), dim=1)
    return mix_conservative_probabilities(
        base_q, conservative_rate, active_light
    )


def _validate_samples(samples: Mapping[str, torch.Tensor], e: torch.Tensor) -> None:
    required = ("f2", "p0", "path_offsets", "eye_index", "light_index", "peak_pdf")
    missing = [name for name in required if name not in samples]
    if missing:
        raise ValueError(f"missing sample arrays: {', '.join(missing)}")

    f2 = samples["f2"]
    p0 = samples["p0"]
    offsets = samples["path_offsets"]
    for name, values in (("E", e), ("f2", f2), ("p0", p0), ("peak_pdf", samples["peak_pdf"])):
        if not torch.all(torch.isfinite(values)).item():
            raise ValueError(f"{name} must contain only finite values")
    for name, values in (("f2", f2), ("p0", p0), ("peak_pdf", samples["peak_pdf"])):
        if torch.any(values < 0).item():
            raise ValueError(f"{name} must be nonnegative")

    node_count = samples["peak_pdf"].numel()
    if f2.ndim != 1 or p0.shape != f2.shape:
        raise ValueError("f2 and p0 must have matching shape [N]")
    if offsets.ndim != 1 or offsets.numel() != f2.numel() + 1:
        raise ValueError("path_offsets must have shape [N + 1]")
    if offsets[0].item() != 0 or offsets[-1].item() != node_count:
        raise ValueError("path_offsets must start at 0 and end at M")
    if torch.any(offsets[1:] < offsets[:-1]).item():
        raise ValueError("path_offsets must be nondecreasing")
    for name in ("eye_index", "light_index", "peak_pdf"):
        if samples[name].ndim != 1 or samples[name].numel() != node_count:
            raise ValueError(f"{name} must have shape [M]")
    if torch.any(samples["eye_index"] < 0).item() or torch.any(
        samples["eye_index"] >= e.shape[0]
    ).item():
        raise ValueError("eye_index is outside E")
    if torch.any(samples["light_index"] < 0).item() or torch.any(
        samples["light_index"] >= e.shape[1]
    ).item():
        raise ValueError("light_index is outside E")


def loss_from_probabilities(
    e: torch.Tensor, samples: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Evaluate the convex objective for a feasible row-probability matrix E."""
    require_torch()
    if e.ndim != 2:
        raise ValueError("E must have shape [num_eye, num_light]")
    _validate_samples(samples, e)

    offsets = samples["path_offsets"]
    counts = offsets[1:] - offsets[:-1]
    path_ids = torch.repeat_interleave(
        torch.arange(samples["f2"].numel(), device=e.device), counts
    )
    node_values = samples["peak_pdf"] * e[
        samples["eye_index"], samples["light_index"]
    ]
    variable_pdf = torch.zeros_like(samples["p0"]).scatter_add_(
        0, path_ids, node_values
    )
    denominator = samples["p0"] + variable_pdf
    if torch.any(denominator <= 0).item():
        raise ValueError("all p0 + A@E denominators must be positive")
    return torch.sum(samples["f2"] / denominator)


def reference_loss(
    logits: torch.Tensor,
    samples: Mapping[str, torch.Tensor],
    conservative_rate: float,
    active_light: torch.Tensor | None = None,
) -> torch.Tensor:
    effective_e = effective_probabilities(
        logits, conservative_rate, active_light
    )
    _validate_samples(samples, effective_e)
    if active_light is not None:
        active = torch.as_tensor(
            active_light, dtype=torch.bool, device=samples["light_index"].device
        )
        light_index = samples["light_index"]
        inactive_nonzero = (
            ~active[light_index]
            & (samples["peak_pdf"] != 0.0)
        )
        if torch.any(inactive_nonzero).item():
            raise ValueError(
                "inactive light columns may only appear on zero-contribution nodes"
            )
    return loss_from_probabilities(effective_e, samples)


def make_synthetic_problem(
    seed: int = 0,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor | float | int]:
    require_torch()
    del seed  # The fixed small problem is deterministic; callers seed random logits.
    dtype = dtype or torch.float64
    torch_device = resolve_device(None if device is None else str(device))
    return {
        "num_eye": 2,
        "num_light": 3,
        "conservative_rate": 0.2,
        "f2": torch.tensor(
            [6.0, 2.0, 1.0, 1.0, 3.0, 8.0], dtype=dtype, device=torch_device
        ),
        "p0": torch.full((6,), 0.5, dtype=dtype, device=torch_device),
        "path_offsets": torch.arange(7, dtype=torch.int64, device=torch_device),
        "eye_index": torch.tensor(
            [0, 0, 0, 1, 1, 1], dtype=torch.int64, device=torch_device
        ),
        "light_index": torch.tensor(
            [0, 1, 2, 0, 1, 2], dtype=torch.int64, device=torch_device
        ),
        "peak_pdf": torch.tensor(
            [5.0, 2.0, 1.0, 1.0, 3.0, 6.0], dtype=dtype, device=torch_device
        ),
    }


def _sample_tensors(problem: Mapping[str, object]) -> dict[str, torch.Tensor]:
    return {
        name: problem[name]
        for name in ("f2", "p0", "path_offsets", "eye_index", "light_index", "peak_pdf")
    }


def check_autograd(
    seed: int = 0, device: str | torch.device | None = None
) -> dict[str, float]:
    require_torch()
    torch_device = resolve_device(None if device is None else str(device))
    torch.manual_seed(seed)
    problem = make_synthetic_problem(dtype=torch.float64, device=torch_device)
    samples = _sample_tensors(problem)
    logits = torch.randn(
        2, 3, dtype=torch.float64, device=torch_device, requires_grad=True
    )
    loss = reference_loss(logits, samples, problem["conservative_rate"])
    (autograd,) = torch.autograd.grad(loss, logits)

    epsilon = 1e-6
    finite_difference = torch.empty_like(logits)
    with torch.no_grad():
        for row in range(logits.shape[0]):
            for column in range(logits.shape[1]):
                plus = logits.detach().clone()
                minus = logits.detach().clone()
                plus[row, column] += epsilon
                minus[row, column] -= epsilon
                finite_difference[row, column] = (
                    reference_loss(plus, samples, problem["conservative_rate"])
                    - reference_loss(minus, samples, problem["conservative_rate"])
                ) / (2.0 * epsilon)

    return {
        "loss": loss.item(),
        "max_abs_error": torch.max(torch.abs(autograd - finite_difference)).item(),
        "device": str(torch_device),
    }


def check_probability_gradient(
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Oracle for production route B: optimize a row-normalized base q directly."""
    require_torch()
    torch_device = resolve_device(None if device is None else str(device))
    dtype = torch.float64
    conservative_rate = 0.2
    samples = {
        "f2": torch.tensor(
            [4.0, 1.0, 3.0, 2.0, 5.0], dtype=dtype, device=torch_device
        ),
        "p0": torch.tensor(
            [0.5, 0.8, 0.4, 0.7, 0.6], dtype=dtype, device=torch_device
        ),
        "path_offsets": torch.tensor(
            [0, 3, 5, 8, 10, 13], dtype=torch.int64, device=torch_device
        ),
        "eye_index": torch.tensor(
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            dtype=torch.int64,
            device=torch_device,
        ),
        "light_index": torch.tensor(
            [0, 2, 1, 3, 0, 1, 3, 2, 1, 2, 0, 3, 3],
            dtype=torch.int64,
            device=torch_device,
        ),
        "peak_pdf": torch.tensor(
            [1.0, 0.5, 0.7, 1.4, 0.8, 0.6, 1.2, 0.9, 1.1, 0.4, 0.3, 1.5, 0.7],
            dtype=dtype,
            device=torch_device,
        ),
    }
    q = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        dtype=dtype,
        device=torch_device,
        requires_grad=True,
    )
    effective = (1.0 - conservative_rate) * q + conservative_rate / q.shape[1]
    loss = loss_from_probabilities(effective, samples)
    (autograd,) = torch.autograd.grad(loss, q)

    offsets = samples["path_offsets"]
    path_ids = torch.repeat_interleave(
        torch.arange(samples["f2"].numel(), device=torch_device),
        offsets[1:] - offsets[:-1],
    )
    node_pdf = samples["peak_pdf"] * effective.detach()[
        samples["eye_index"], samples["light_index"]
    ]
    variable_pdf = torch.zeros_like(samples["p0"]).scatter_add_(0, path_ids, node_pdf)
    path_gradient = -samples["f2"] / (samples["p0"] + variable_pdf).square()
    node_gradient = (
        (1.0 - conservative_rate)
        * samples["peak_pdf"]
        * path_gradient[path_ids]
    )
    flat_index = samples["eye_index"] * q.shape[1] + samples["light_index"]
    analytic = torch.zeros_like(q).view(-1).scatter_add_(0, flat_index, node_gradient)
    analytic = analytic.view_as(q)

    return {
        "loss": loss.item(),
        "max_abs_error": torch.max(torch.abs(autograd - analytic)).item(),
        "min_q": q.detach().min().item(),
        "max_row_sum_error": torch.max(
            torch.abs(q.detach().sum(dim=1) - 1.0)
        ).item(),
        "device": str(torch_device),
    }


def check_convexity(
    seed: int = 0, device: str | torch.device | None = None
) -> dict[str, float]:
    require_torch()
    torch_device = resolve_device(None if device is None else str(device))
    torch.manual_seed(seed)
    problem = make_synthetic_problem(dtype=torch.float64, device=torch_device)
    samples = _sample_tensors(problem)
    rate = problem["conservative_rate"]
    e_a = effective_probabilities(
        torch.randn(2, 3, dtype=torch.float64, device=torch_device), rate
    )
    e_b = effective_probabilities(
        torch.randn(2, 3, dtype=torch.float64, device=torch_device), rate
    )
    mix = 0.37
    mixed_e = mix * e_a + (1.0 - mix) * e_b
    lhs = loss_from_probabilities(mixed_e, samples)
    rhs = (
        mix * loss_from_probabilities(e_a, samples)
        + (1.0 - mix) * loss_from_probabilities(e_b, samples)
    )
    return {
        "jensen_gap": (lhs - rhs).item(),
        "lhs": lhs.item(),
        "rhs": rhs.item(),
        "device": str(torch_device),
    }


def optimize(
    samples: Mapping[str, torch.Tensor],
    num_eye: int,
    num_light: int,
    conservative_rate: float,
    steps: int = 200,
    learning_rate: float = 0.05,
    active_light: torch.Tensor | None = None,
    initial_base_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float]]:
    require_torch()
    if steps <= 0:
        raise ValueError("steps must be positive")
    if initial_base_q is None:
        initial_logits = torch.zeros(
            num_eye,
            num_light,
            dtype=samples["f2"].dtype,
            device=samples["f2"].device,
        )
    else:
        initial_q = torch.as_tensor(
            initial_base_q,
            dtype=samples["f2"].dtype,
            device=samples["f2"].device,
        )
        if initial_q.shape != (num_eye, num_light):
            raise ValueError("initial_base_q has the wrong shape")
        initial_q = mix_conservative_probabilities(
            initial_q, 0.0, active_light
        )
        initial_logits = torch.log(
            torch.clamp_min(initial_q, torch.finfo(initial_q.dtype).tiny)
        )
    logits = initial_logits.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = reference_loss(
            logits, samples, conservative_rate, active_light
        )
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    with torch.no_grad():
        if active_light is None:
            base_q = torch.softmax(logits, dim=1)
        else:
            active = torch.as_tensor(
                active_light, dtype=torch.bool, device=logits.device
            )
            base_q = torch.softmax(
                logits.masked_fill(~active, -torch.inf), dim=1
            )
        e = mix_conservative_probabilities(
            base_q, conservative_rate, active_light
        )
        history.append(loss_from_probabilities(e, samples).item())
    return base_q, history


def check_loss_descent(
    seed: int = 0,
    steps: int = 150,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    require_torch()
    torch_device = resolve_device(None if device is None else str(device))
    torch.manual_seed(seed)
    problem = make_synthetic_problem(dtype=torch.float64, device=torch_device)
    _, history = optimize(
        _sample_tensors(problem),
        problem["num_eye"],
        problem["num_light"],
        problem["conservative_rate"],
        steps=steps,
    )
    return {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "device": str(torch_device),
    }


def _load_binary_snapshot(path: Path, device: str) -> dict[str, object]:
    import struct

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to read optimal-E snapshots") from exc

    common_header_format = "<8sIIQQIII"
    common_header_size = struct.calcsize(common_header_format)
    with path.open("rb") as stream:
        header = stream.read(common_header_size)
        if len(header) != common_header_size:
            raise ValueError("truncated optimal-E snapshot header")
        (
            magic,
            schema_version,
            endian_marker,
            num_paths,
            num_nodes,
            num_eye,
            num_light,
            active_light_count,
        ) = struct.unpack(common_header_format, header)
        if magic != b"SPCBE001" or schema_version not in (1, 2):
            raise ValueError("unsupported optimal-E binary snapshot")
        if endian_marker != 0x01020304:
            raise ValueError("optimal-E snapshot byte order is unsupported")
        experiment_seed = 0
        if schema_version == 2:
            experiment_seed_data = stream.read(4)
            if len(experiment_seed_data) != 4:
                raise ValueError("truncated optimal-E snapshot header")
            experiment_seed = struct.unpack("<I", experiment_seed_data)[0]
        conservative_rate_data = stream.read(4)
        if len(conservative_rate_data) != 4:
            raise ValueError("truncated optimal-E snapshot header")
        conservative_rate = struct.unpack("<f", conservative_rate_data)[0]
        header_size = common_header_size + 4 + (
            4 if schema_version == 2 else 0
        )
        if min(num_paths, num_eye, num_light, active_light_count) <= 0:
            raise ValueError("optimal-E snapshot dimensions must be positive")
        matrix_size = num_eye * num_light
        expected_size = header_size + 4 * (
            2 * num_paths
            + 2 * num_nodes
            + num_paths + 1
            + num_light
            + matrix_size
        )
        if path.stat().st_size != expected_size:
            raise ValueError("optimal-E snapshot size does not match its header")

        def read_array(dtype: str, count: int):
            values = np.fromfile(stream, dtype=dtype, count=count)
            if values.size != count:
                raise ValueError("truncated optimal-E snapshot array")
            return values

        f2 = read_array("<f4", num_paths)
        p0 = read_array("<f4", num_paths)
        peak_pdf = read_array("<f4", num_nodes)
        path_offsets = read_array("<i4", num_paths + 1)
        matrix_indices = read_array("<i4", num_nodes)
        active_light = read_array("<i4", num_light)
        base_q = read_array("<f4", matrix_size).reshape(num_eye, num_light)

    if int(active_light.sum()) != active_light_count:
        raise ValueError("optimal-E active-light count is inconsistent")
    problem = {
        "num_eye": int(num_eye),
        "num_light": int(num_light),
        "conservative_rate": float(conservative_rate),
        "experiment_seed": int(experiment_seed),
        "f2": torch.as_tensor(f2, dtype=torch.float64, device=device),
        "p0": torch.as_tensor(p0, dtype=torch.float64, device=device),
        "peak_pdf": torch.as_tensor(peak_pdf, dtype=torch.float64, device=device),
        "path_offsets": torch.as_tensor(
            path_offsets, dtype=torch.int64, device=device
        ),
        "eye_index": torch.as_tensor(
            matrix_indices // num_light, dtype=torch.int64, device=device
        ),
        "light_index": torch.as_tensor(
            matrix_indices % num_light, dtype=torch.int64, device=device
        ),
        "active_light": torch.as_tensor(
            active_light, dtype=torch.bool, device=device
        ),
        "base_q": torch.as_tensor(base_q, dtype=torch.float64, device=device),
    }
    return problem


def load_export(path: Path, device: str) -> dict[str, object]:
    require_torch()
    with path.open("rb") as stream:
        if stream.read(8) == b"SPCBE001":
            return _load_binary_snapshot(path, device)
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to read the documented .npz format") from exc

    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "num_eye",
            "num_light",
            "conservative_rate",
            "f2",
            "p0",
            "path_offsets",
            "eye_index",
            "light_index",
            "peak_pdf",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"missing export arrays: {', '.join(missing)}")
        if int(data["schema_version"].item()) != 1:
            raise ValueError("unsupported optimal-E export schema_version")
        problem = {
            "num_eye": int(data["num_eye"]),
            "num_light": int(data["num_light"]),
            "conservative_rate": float(data["conservative_rate"]),
        }
        for name in ("f2", "p0", "peak_pdf", "base_q"):
            if name in data.files:
                problem[name] = torch.as_tensor(
                    data[name], dtype=torch.float64, device=device
                )
        for name in ("path_offsets", "eye_index", "light_index"):
            problem[name] = torch.as_tensor(
                data[name], dtype=torch.int64, device=device
            )
        if "active_light" in data.files:
            problem["active_light"] = torch.as_tensor(
                data["active_light"], dtype=torch.bool, device=device
            )
    return problem


def load_cuda_result(path: Path, device: str) -> dict[str, object]:
    import struct

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required to read CUDA validation results") from exc

    header_format = "<8sIIIIIff"
    header_size = struct.calcsize(header_format)
    with path.open("rb") as stream:
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError("truncated CUDA validation result header")
        (
            magic,
            schema_version,
            endian_marker,
            num_eye,
            num_light,
            accepted_steps,
            initial_objective,
            final_objective,
        ) = struct.unpack(header_format, header)
        if magic != b"SPCBR001" or schema_version != 1:
            raise ValueError("unsupported CUDA validation result")
        if endian_marker != 0x01020304:
            raise ValueError("CUDA validation result byte order is unsupported")
        matrix_size = num_eye * num_light
        if min(num_eye, num_light) <= 0:
            raise ValueError("CUDA validation result dimensions must be positive")
        if path.stat().st_size != header_size + 8 * matrix_size:
            raise ValueError("CUDA validation result size does not match its header")
        initial_gradient = np.fromfile(stream, dtype="<f4", count=matrix_size)
        final_q = np.fromfile(stream, dtype="<f4", count=matrix_size)
    return {
        "num_eye": int(num_eye),
        "num_light": int(num_light),
        "accepted_steps": int(accepted_steps),
        "initial_objective": float(initial_objective),
        "final_objective": float(final_objective),
        "initial_gradient": torch.as_tensor(
            initial_gradient.reshape(num_eye, num_light),
            dtype=torch.float64,
            device=device,
        ),
        "final_q": torch.as_tensor(
            final_q.reshape(num_eye, num_light),
            dtype=torch.float64,
            device=device,
        ),
    }


def compare_cuda_result(
    problem: Mapping[str, object],
    cuda_result: Mapping[str, object],
    objective_rtol: float = 2e-5,
    objective_atol: float = 1e-4,
    gradient_rtol: float = 5e-3,
    gradient_atol: float = 1e-4,
) -> dict[str, float]:
    require_torch()
    if "base_q" not in problem:
        raise ValueError("cross-check input must contain base_q")
    if (
        cuda_result["num_eye"] != problem["num_eye"]
        or cuda_result["num_light"] != problem["num_light"]
    ):
        raise ValueError("CUDA result dimensions disagree with the snapshot")

    base_q = problem["base_q"].detach().clone().requires_grad_(True)
    active = problem.get("active_light")
    if active is None:
        active = torch.ones(
            problem["num_light"], dtype=torch.bool, device=base_q.device
        )
    active_float = active.to(dtype=base_q.dtype)
    if not torch.allclose(
        (base_q * active_float).sum(dim=1),
        torch.ones(problem["num_eye"], dtype=base_q.dtype, device=base_q.device),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("snapshot base_q rows must sum to one on active columns")
    uniform = active_float / active_float.sum()
    initial_e = (
        (1.0 - problem["conservative_rate"]) * base_q
        + problem["conservative_rate"] * uniform
    ) * active_float
    initial_loss = loss_from_probabilities(initial_e, _sample_tensors(problem))
    (initial_gradient,) = torch.autograd.grad(initial_loss, base_q)

    cuda_initial = float(cuda_result["initial_objective"])
    initial_abs_error = abs(initial_loss.item() - cuda_initial)
    initial_tolerance = objective_atol + objective_rtol * abs(initial_loss.item())
    if initial_abs_error > initial_tolerance:
        raise AssertionError(
            "CUDA initial objective disagrees with PyTorch: "
            f"abs_error={initial_abs_error:.6g}, tolerance={initial_tolerance:.6g}"
        )

    cuda_gradient = cuda_result["initial_gradient"]
    gradient_diff = torch.abs(initial_gradient - cuda_gradient)
    gradient_tolerance = gradient_atol + gradient_rtol * torch.abs(initial_gradient)
    if not torch.all(gradient_diff <= gradient_tolerance).item():
        raise AssertionError(
            "CUDA gradient disagrees with PyTorch: "
            f"max_abs_error={gradient_diff.max().item():.6g}"
        )

    final_q = cuda_result["final_q"]
    final_e = (
        (1.0 - problem["conservative_rate"]) * final_q
        + problem["conservative_rate"] * uniform
    ) * active_float
    final_loss = loss_from_probabilities(final_e, _sample_tensors(problem))
    cuda_final = float(cuda_result["final_objective"])
    final_abs_error = abs(final_loss.item() - cuda_final)
    final_tolerance = objective_atol + objective_rtol * abs(final_loss.item())
    if final_abs_error > final_tolerance:
        raise AssertionError(
            "CUDA final objective disagrees with PyTorch: "
            f"abs_error={final_abs_error:.6g}, tolerance={final_tolerance:.6g}"
        )

    return {
        "initial_objective_abs_error": initial_abs_error,
        "gradient_max_abs_error": gradient_diff.max().item(),
        "final_objective_abs_error": final_abs_error,
        "cuda_initial_objective": cuda_initial,
        "cuda_final_objective": cuda_final,
        "accepted_steps": float(cuda_result["accepted_steps"]),
    }


def run_self_test(device: str | torch.device | None = None) -> dict[str, dict[str, float]]:
    torch_device = resolve_device(None if device is None else str(device))
    results = {
        "autograd": check_autograd(seed=7, device=torch_device),
        "probability_gradient": check_probability_gradient(device=torch_device),
        "convexity": check_convexity(seed=11, device=torch_device),
        "descent": check_loss_descent(seed=13, device=torch_device),
    }
    if results["autograd"]["max_abs_error"] >= 2e-6:
        raise AssertionError(f"autograd check failed: {results['autograd']}")
    if results["probability_gradient"]["max_abs_error"] >= 1e-12:
        raise AssertionError(
            f"probability gradient check failed: {results['probability_gradient']}"
        )
    if results["convexity"]["jensen_gap"] > 1e-10:
        raise AssertionError(f"convexity check failed: {results['convexity']}")
    if results["descent"]["final_loss"] >= results["descent"]["initial_loss"] * 0.95:
        raise AssertionError(f"loss descent check failed: {results['descent']}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--describe-format", action="store_true")
    parser.add_argument(
        "--input",
        type=Path,
        help="real-sample export in .spcoe binary or documented .npz format",
    )
    parser.add_argument(
        "--cuda-result",
        type=Path,
        help="production CUDA result (.spcor) for direct cross-check",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional .npy output for optimized base q (before conservative mixing)",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="optional little-endian float32 base q for the CUDA evaluator",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="optional JSON summary for batch experiments",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: auto (cuda if available else cpu), cpu, cuda, or cuda:N",
    )
    args = parser.parse_args(argv)

    if args.describe_format:
        print(EXPORT_FORMAT)
        if args.input is None:
            return 0
    try:
        require_torch()
        torch_device = resolve_device(args.device)
        print(f"device: {torch_device}")
        if args.input is None:
            results = run_self_test(device=torch_device)
            for name, values in results.items():
                print(f"{name}: {values}")
            print("SELF_TEST_OK")
            return 0

        problem = load_export(args.input, str(torch_device))
        cross_check_metrics = None
        if args.cuda_result is not None:
            cuda_result = load_cuda_result(args.cuda_result, str(torch_device))
            cross_check_metrics = compare_cuda_result(problem, cuda_result)
            print(f"cross_check: {cross_check_metrics}")
            print("CROSS_CHECK_OK")
        base_q, history = optimize(
            _sample_tensors(problem),
            problem["num_eye"],
            problem["num_light"],
            problem["conservative_rate"],
            steps=args.steps,
            learning_rate=args.learning_rate,
            active_light=problem.get("active_light"),
            initial_base_q=problem.get("base_q"),
        )
        print(f"loss: {history[0]:.12g} -> {history[-1]:.12g}")
        if args.output is not None or args.raw_output is not None:
            import numpy as np

            host_q = base_q.detach().cpu().numpy()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output, host_q)
            print(f"saved base q to {args.output}")
        if args.raw_output is not None:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            np.asarray(host_q, dtype="<f4").tofile(args.raw_output)
            print(f"saved float32 base q to {args.raw_output}")
        if args.metrics_output is not None:
            report = {
                "device": str(torch_device),
                "experiment_seed": int(problem.get("experiment_seed", 0)),
                "pytorch": {
                    "steps": args.steps,
                    "learning_rate": args.learning_rate,
                    "initial_loss": history[0],
                    "final_loss": history[-1],
                },
            }
            if cross_check_metrics is not None:
                report["cross_check"] = cross_check_metrics
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"saved metrics to {args.metrics_output}")
        return 0
    except (RuntimeError, ValueError, AssertionError, OSError) as exc:
        print(f"optimal_e_reference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
