"""Optimise three objectives along a solution manifold."""

from pathlib import Path

from quantum_splitting import optimize, save_formula, search

ORDER = 4
M = 12  # Six free complex coefficients for two complex conditions.
SEED = 20
SEARCH_ATTEMPTS = 100
WORKERS = 1
DIGITS = 64
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    found = search(
        order=ORDER,
        m=M,
        seed=SEED,
        attempts=SEARCH_ATTEMPTS,
        workers=WORKERS,
        tolerance=10.0 ** (1 - DIGITS),
    )
    if not found.converged:
        raise SystemExit(
            f"No verified order-{ORDER} formula. Starts tried {found.attempts}. "
            f"{found.residual_label.capitalize()} {found.residual:.3e}."
        )

    formula = found.formula
    # Start every objective from the same verified formula.
    for objective in ("l1", "l2", "error"):
        print(f"\nOptimising {objective}", flush=True)
        result = optimize(formula, objective=objective, digits=DIGITS)
        path = OUTPUT_DIR / f"order{formula.order}_m{formula.m}_min{objective}.json"
        save_formula(result.formula, path, dps=DIGITS + 20, seed=SEED)
        print(f"  Objective  {result.initial_value:.10g} → {result.value:.10g}")
        print(f"  {result.message}")
        print(f"  Saved {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
