from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_gofr(x: np.ndarray, g: np.ndarray, title: str, out: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, g, label="g(r)")
    ax.set_xlabel(r"r $\AA$")
    ax.set_ylabel(r"g(r)")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pmf(x: np.ndarray, v: np.ndarray, title: str, out: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, v)
    ax.set_xlabel(r"r $\AA$")
    ax.set_ylabel(r"v(r) (dimensionless)")
    ax.set_title(title)
    ax.grid(True, which="both")

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


