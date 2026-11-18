"""Command-line tools for searching and working with coefficient files."""

from __future__ import annotations

import argparse
import sys
from math import ceil, log10
from pathlib import Path
from typing import Sequence

from mpmath import nstr

from .coefficients import load_formula, save_formula
from .model import Formula
from .optimize import OptimizationError, optimize
from .precision import RefinementError, refine, verify
from .search import _DEFAULT_MAX_NFEV, SearchResult, search


def _add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, required=True, help="output JSON file")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-splitting",
        description="Search, optimise, and verify splitting coefficients for dissipative dynamics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    search_parser = commands.add_parser("search", help="run a multistart coefficient search")
    search_parser.add_argument("--order", type=int, required=True, help="target even order")
    search_parser.add_argument("--m", type=int, required=True, help="number of equal A factors")
    search_parser.add_argument("--attempts", type=int, default=20, help="maximum starts (20)")
    search_parser.add_argument("--seed", type=int, default=0, help="random seed (0)")
    search_parser.add_argument("--initial", type=Path, help="optional starting formula in JSON")
    search_parser.add_argument("--min-real", default="0", help="strict lower bound on Re(b) (0)")
    search_parser.add_argument(
        "--max-nfev",
        type=int,
        default=_DEFAULT_MAX_NFEV,
        help="residual evaluation budget per start (%(default)s)",
    )
    search_parser.add_argument(
        "--tolerance", type=float, default=1e-12, help="maximum full-word residual (1e-12)"
    )
    search_parser.add_argument("--workers", type=int, default=1, help="parallel processes (1)")
    _add_output(search_parser)

    verify_parser = commands.add_parser("verify", help="check all word coefficients")
    verify_parser.add_argument("formula", type=Path, help="path to a coefficient JSON file")
    verify_parser.add_argument("--dps", type=int, default=80, help="decimal working precision (80)")
    verify_parser.add_argument(
        "--tolerance", default="1e-12", help="maximum full-word residual (1e-12)"
    )
    verify_parser.add_argument(
        "--leading", action="store_true", help="also report the residual at order + 1"
    )

    refine_parser = commands.add_parser("refine", help="change the stored coefficient precision")
    refine_parser.add_argument("formula", type=Path, help="path to a coefficient JSON file")
    refine_parser.add_argument(
        "--digits",
        type=int,
        required=True,
        help="significant figures per component (lower values truncate)",
    )
    refine_parser.add_argument(
        "--max-iterations", type=int, default=12, help="Newton iteration limit (12)"
    )
    _add_output(refine_parser)

    optimize_parser = commands.add_parser(
        "optimize", help="improve an objective along the manifold of solutions"
    )
    optimize_parser.add_argument("formula", type=Path, help="path to a coefficient JSON file")
    optimize_parser.add_argument(
        "--objective",
        choices=("l1", "l2", "error"),
        default="l1",
        help="objective to minimize (l1)",
    )
    optimize_parser.add_argument("--min-real", default="0", help="strict lower bound on Re(b) (0)")
    optimize_parser.add_argument(
        "--digits", type=int, default=50, help="significant figures after refinement (50)"
    )
    optimize_parser.add_argument(
        "--max-iterations", type=int, help="optional limit on local solver iterations"
    )
    _add_output(optimize_parser)
    return parser


def _search(args: argparse.Namespace) -> int:
    initial = load_formula(args.initial) if args.initial is not None else None
    seed = args.seed if initial is None else None

    def save_best(candidate: SearchResult) -> None:
        save_formula(candidate.formula, args.output, include_diagnostics=False, seed=seed)

    result = search(
        args.order,
        m=args.m,
        attempts=args.attempts,
        seed=args.seed,
        initial=initial,
        min_real=args.min_real,
        max_nfev=args.max_nfev,
        tolerance=args.tolerance,
        workers=args.workers,
        on_improvement=save_best,
    )
    save_formula(
        result.formula, args.output, dps=max(80, 20 + ceil(-log10(args.tolerance))), seed=seed
    )
    print(
        f"Search {'converged' if result.converged else 'exhausted'} after {result.attempts} starts"
    )
    label = "Maximum residual" if result.verified else "Best residual (at least)"
    print(f"{label} through order {args.order} = {result.residual:.6e}")
    print(f"Saved {args.output}")
    return 0 if result.converged else 1


def _verify(args: argparse.Namespace, formula: Formula) -> int:
    result = verify(
        formula,
        dps=args.dps,
        tolerance=args.tolerance,
        include_leading=args.leading,
    )
    print(f"Order {result.order}, m={formula.m}, {result.dps} working digits")
    print(f"Minimum Re(b) = {nstr(result.min_real_b, 10)}")
    for degree, residual in enumerate(result.residuals_by_degree, start=1):
        label = " (leading error diagnostic)" if degree > result.order else ""
        print(f"Degree {degree} residual = {nstr(residual, 10)}{label}")
    if result.leading_l2 is not None:
        print(f"Degree {result.order + 1} word L2 error = {nstr(result.leading_l2, 10)}")
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status}  maximum residual {nstr(result.max_residual, 10)}"
        f"  tolerance {nstr(result.tolerance, 10)}"
    )
    return 0 if result.passed else 1


def _optimize(args: argparse.Namespace, formula: Formula) -> int:
    result = optimize(
        formula,
        objective=args.objective,
        min_real=args.min_real,
        digits=args.digits,
        max_iterations=args.max_iterations,
    )
    save_formula(result.formula, args.output, dps=max(80, args.digits + 20))
    print(f"{result.objective}  {result.initial_value:.10g} -> {result.value:.10g}")
    residual = nstr(result.residual, 7, min_fixed=0, max_fixed=0)
    print(f"Maximum residual through order {formula.order} = {residual}")
    print(result.message)
    print(f"Saved {args.output}")
    return 0 if result.converged else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its process status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            return _search(args)

        formula = load_formula(args.formula)
        if args.command == "verify":
            return _verify(args, formula)
        if args.command == "optimize":
            return _optimize(args, formula)
        formula = refine(formula, digits=args.digits, max_iterations=args.max_iterations)
        save_formula(formula, args.output, dps=max(80, args.digits + 20))
        print(f"Saved {args.output}")
        return 0
    except (ValueError, OSError, RefinementError, OptimizationError) as error:
        print(f"{parser.prog}: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
