"""Exact real-part bounds for coefficient strings."""

from fractions import Fraction

from .model import Formula


def real_floor(value: float | str, m: int) -> Fraction:
    """Interpret a real-part floor as written and check its feasibility."""
    message = "min_real must be nonnegative and below 1/(m + 1)"
    try:
        floor = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError(message) from None
    if not 0 <= floor < Fraction(1, m + 1):
        raise ValueError(message)
    return floor


def above_real_floor(formula: Formula, floor: Fraction) -> bool:
    """Check whether every stored real coefficient strictly exceeds the floor."""
    return all(Fraction(real) > floor for real, _ in formula.b_half)
