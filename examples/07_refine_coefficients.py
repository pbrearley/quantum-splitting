"""Refine an archived formula and verify the extra digits."""

from pathlib import Path

from mpmath import nstr

from quantum_splitting import RefinementError, load_formula, refine, save_formula, verify

EXAMPLES = Path(__file__).resolve().parent
SOURCE = EXAMPLES.parent / "coefficients" / "order6_m16.json"
DIGITS = 100


def main() -> None:
    formula = load_formula(SOURCE)
    refined = refine(formula, digits=DIGITS)
    dps = DIGITS + 20
    tolerance = f"1e{1 - DIGITS}"
    report = verify(refined, dps=dps, tolerance=tolerance)
    residual = nstr(report.max_residual, 8)
    if not report.passed:
        raise SystemExit(f"Verification failed at tolerance {tolerance}. Max residual {residual}.")

    destination = EXAMPLES / f"{SOURCE.stem}_{DIGITS}digits.json"
    save_formula(refined, destination, dps=dps)
    print(f"Refined {SOURCE.name} to {DIGITS} significant figures.")
    print(f"PASS  max residual {residual}  tolerance {tolerance}")
    print(f"  Saved {destination}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RefinementError, ValueError) as error:
        raise SystemExit(str(error)) from None
