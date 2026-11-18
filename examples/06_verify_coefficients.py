"""Verify all example coefficient files against the order conditions."""

from pathlib import Path

from mpmath import nstr

from quantum_splitting import load_formula, verify

COEFFICIENTS = Path(__file__).resolve().parents[1] / "coefficients"
FORMULAS = (
    ("order4_m4.json", 4, 4),
    ("order6_m16.json", 6, 16),
    ("order6_m16_minl1.json", 6, 16),
    ("order8_m52.json", 8, 52),
    ("order8_m52_minl1.json", 8, 52),
    ("order10_m164.json", 10, 164),
    ("order10_m164_minl1.json", 10, 164),
)
DPS = 84
TOLERANCE = "1e-63"


def main() -> int:
    all_passed = True
    for filename, order, m in FORMULAS:
        path = COEFFICIENTS / filename
        try:
            formula = load_formula(path)
            if (formula.order, formula.m) != (order, m):
                raise ValueError(f"expected order {order} and m={m}")
            report = verify(formula, dps=DPS, tolerance=TOLERANCE)
        except (OSError, ValueError) as error:
            print(f"FAIL  {path.name}  {error}")
            all_passed = False
            continue

        status = "PASS" if report.passed else "FAIL"
        print(f"{status}  {path.name}  max residual {nstr(report.max_residual, 8)}")
        all_passed &= report.passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
