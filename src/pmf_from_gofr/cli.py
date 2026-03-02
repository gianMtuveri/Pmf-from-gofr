from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .core import PMFParams, compute_pmf_from_file
from .plotting import plot_gofr, plot_pmf


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute PMF-like curve from gofr .dat files with tail normalization of g(r)."
    )
    p.add_argument("--names", nargs="+", required=True, help="Base filenames without .dat")
    p.add_argument("--in-dir", type=Path, default=Path("../"), help="Directory containing *.dat")
    p.add_argument("--out-dir", type=Path, default=Path("./"), help="Directory for outputs")

    p.add_argument("--plots", action="store_true", help="Save plots (g(r) and v(r)) as PNGs")
    p.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")

    # Tail normalization options
    p.add_argument("--tail-start", type=float, default=None, help="Tail start in Å (e.g. 30). If omitted uses tail-fraction.")
    p.add_argument("--tail-fraction", type=float, default=0.2, help="Tail fraction used if tail-start is omitted (default 0.2).")
    p.add_argument("--tail-method", choices=["scale", "shift"], default="scale", help="Tail correction method (default scale).")

    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    log = logging.getLogger("pmf_from_gofr")

    params = PMFParams(
        tail_start=args.tail_start,
        tail_fraction=args.tail_fraction,
        tail_method=args.tail_method,
    )

    for name in args.names:
        dat_path = args.in_dir / f"{name}.dat"
        if not dat_path.is_file():
            log.warning("%s not found", dat_path)
            continue

        out_xvg = args.out_dir / f"PMF{name}.xvg"
        result = compute_pmf_from_file(dat_path, out_xvg, params)
        log.info("%s: converged in n=%d iterations -> %s", name, result["n"], out_xvg)

        if args.plots:
            plot_gofr(result["x"], result["g"], title=name, out=args.out_dir / f"gofr_{name}.png")
            plot_pmf(result["x"], result["v"], title=name, out=args.out_dir / f"PMF{name}.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

