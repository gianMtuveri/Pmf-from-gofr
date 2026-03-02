from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def savitzky_golay(y: np.ndarray, window_size: int, order: int, deriv: int = 0, rate: int = 1) -> np.ndarray:
    """Savitzky–Golay smoothing (and optional derivative).

    Kept dependency-free to match typical HPC environments.
    """
    from math import factorial

    y = np.asarray(y, dtype=float)

    window_size = abs(int(window_size))
    order = abs(int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise ValueError("window_size must be a positive odd integer")
    if window_size < order + 2:
        raise ValueError("window_size is too small for the polynomial order")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    b = np.array([[k**i for i in order_range] for k in range(-half_window, half_window + 1)], dtype=float)
    m = np.linalg.pinv(b)[deriv] * (rate**deriv) * factorial(deriv)

    # pad
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    ypad = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], ypad, mode="valid")


@dataclass(frozen=True)
class PMFParams:
    """Parameters controlling preprocessing + iterative closure."""

    kT: float = 0.592

    # LJ initial guess
    eps: float = 4.0
    sigma: float = 4.0

    # S-G smoothing
    sg_window: int = 15
    sg_order: int = 2

    # Tail normalization for g(r) -> 1
    # If tail_start is None, uses last `tail_fraction` of points.
    tail_start: float | None = None
    tail_fraction: float = 0.2
    tail_method: str = "scale"  # "scale" recommended; "shift" available

    # Numerical safety for log(g)
    min_g: float = 1e-12

    # Iteration control
    conv_tol: float = 1e-5
    max_iter: int = 10_000


def load_gofr_dat(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, g, m from a 3-column gofr .dat file.

    Expected columns:
        x   g(r)   m(r)  (third column kept for completeness; not used by default)
    """
    data = np.loadtxt(path, comments=("@", "#"))
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} must have at least 3 columns (x, g, m). Got shape {data.shape}.")
    x = data[:, 0].astype(float)
    g = data[:, 1].astype(float)
    m = data[:, 2].astype(float)
    return x, g, m


def trim_to_positive_g(x: np.ndarray, g: np.ndarray, *, min_g: float) -> tuple[np.ndarray, np.ndarray]:
    """Trim leading region until g > min_g.

    This handles initial zeros commonly present at small r in RDF outputs.
    """
    x = np.asarray(x, float)
    g = np.asarray(g, float)

    idx = np.where(g > min_g)[0]
    if idx.size == 0:
        raise ValueError("g(r) is never positive; cannot proceed.")
    start = int(idx[0])
    return x[start:], g[start:]


def normalize_g_tail(
    x: np.ndarray,
    g: np.ndarray,
    *,
    tail_start: float | None = None,
    tail_fraction: float = 0.2,
    method: str = "scale",
    target: float = 1.0,
) -> np.ndarray:
    """Normalize the long-range tail so that g(r) approaches `target` (default 1.0).

    Parameters
    ----------
    x, g
        Arrays for distance and RDF.
    tail_start
        If provided, uses x >= tail_start as the tail region.
    tail_fraction
        If tail_start is None, uses the last `tail_fraction` of points.
    method
        - "scale": g *= target / median(g_tail)  (recommended; preserves positivity)
        - "shift": g -= (median(g_tail) - target)
    """
    x = np.asarray(x, float)
    g = np.asarray(g, float)

    if tail_start is not None:
        mask = x >= float(tail_start)
    else:
        n = len(x)
        k = max(1, int(round(n * float(tail_fraction))))
        mask = np.zeros(n, dtype=bool)
        mask[-k:] = True

    g_tail = g[mask]
    g_tail = g_tail[np.isfinite(g_tail)]

    # Not enough tail points -> leave unchanged
    if g_tail.size < 5:
        return g

    baseline = float(np.median(g_tail))

    if method == "scale":
        if baseline == 0.0:
            return g
        return g * (target / baseline)

    if method == "shift":
        return g - (baseline - target)

    raise ValueError("method must be 'scale' or 'shift'")


def lennard_jones_u(x: np.ndarray, eps: float, sigma: float) -> np.ndarray:
    """Lennard–Jones potential used as initial u(r)."""
    x = np.asarray(x, dtype=float)
    return 4.0 * eps * ((sigma / x) ** 12 - (sigma / x) ** 6)


def iterate_closure(x: np.ndarray, g: np.ndarray, params: PMFParams) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the iterative closure scheme.

    Returns
    -------
    C_last : np.ndarray
        Last computed C(r).
    u_last : np.ndarray
        Last computed u(r).
    n : int
        Number of iterations.
    """
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)

    h = g - 1.0

    # Use LJ as initial guess; keep two previous states for convergence check
    u_prev2 = lennard_jones_u(x, params.eps, params.sigma)
    u_prev1 = u_prev2.copy()

    n = 0
    cond = np.inf
    C_last = np.zeros_like(g)

    while cond > params.conv_tol and n < params.max_iter:
        # C(u) = g - g * exp(-u/kT)
        C_last = g - g * np.exp(-u_prev1 / params.kT)

        # u_new = kT * (h - log(g) - C)
        u_new = params.kT * (h - np.log(g) - C_last)

        # convergence metric similar in spirit to your u[-1] vs u[-3]
        cond = float(np.sum(np.abs(u_new - u_prev2)))

        u_prev2, u_prev1 = u_prev1, u_new
        n += 1

    return C_last, u_prev1, n


def compute_pmf_from_file(dat_path: Path, out_xvg_path: Path, params: PMFParams) -> dict:
    """Full pipeline for one .dat file: load -> preprocess -> iterate -> save PMF xvg.

    Outputs a two-column XVG-like text file:
        x   v(x)
    where:
        v = h - log(g) - C_last

    Notes
    -----
    - No additional shift is applied to v(r) (per user preference).
    """
    x, g, m = load_gofr_dat(dat_path)

    # trim zeros / non-positive values
    x, g = trim_to_positive_g(x, g, min_g=params.min_g)

    # optional smoothing
    g = savitzky_golay(g, params.sg_window, params.sg_order)

    # enforce tail normalization (addresses long-r drift)
    g = normalize_g_tail(
        x,
        g,
        tail_start=params.tail_start,
        tail_fraction=params.tail_fraction,
        method=params.tail_method,
        target=1.0,
    )

    # ensure log-safe
    g = np.clip(g, params.min_g, None)

    # iterate closure
    C, u, n = iterate_closure(x, g, params)

    h = g - 1.0
    v = h - np.log(g) - C  # saved output (no tail shift)

    out_xvg_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_xvg_path, np.column_stack([x, v]))

    return {
        "x": x,
        "g": g,
        "h": h,
        "C": C,
        "u": u,
        "v": v,
        "n": n,
        "input": str(dat_path),
        "output": str(out_xvg_path),
    }


def build_input_paths(names: Iterable[str], base_dir: Path) -> list[Path]:
    """Helper to build paths like base_dir/name.dat"""
    return [base_dir / f"{name}.dat" for name in names]

