"""Search for an order-10 splitting and save the resulting coefficients."""

from pathlib import Path

from quantum_splitting import save_formula, search

ORDER = 10
M = 164
SEED = 1342945
SEARCH_ATTEMPTS = 100
WORKERS = 1
MIN_REAL = 1e-10
DIGITS = 64
OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    found = search(
        order=ORDER,
        m=M,
        seed=SEED,
        attempts=SEARCH_ATTEMPTS,
        workers=WORKERS,
        min_real=MIN_REAL,
        tolerance=10.0 ** (1 - DIGITS),
    )
    if not found.converged:
        raise SystemExit(
            f"No verified order-{ORDER} formula. Starts tried {found.attempts}. "
            f"{found.residual_label.capitalize()} {found.residual:.3e}."
        )
    print(f"Found a verified formula on start {found.attempts}.", flush=True)

    formula = found.formula
    path = OUTPUT_DIR / f"order{formula.order}_m{formula.m}.json"
    save_formula(formula, path, dps=DIGITS + 20, seed=SEED)
    print(f"  Saved {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from None
