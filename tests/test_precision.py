"""Order identities, independent verification and coefficient precision."""

from fractions import Fraction
from pathlib import Path

import pytest
from mpmath import mp

from quantum_splitting import Formula, load_formula, refine, verify


def test_strang_leading_word_error_matches_analytic_value():
    # The ABA coefficient is zero, the BBA and ABB coefficients are 1/8,
    # and the AAB, BAA and BAB coefficients are 1/4. The maximum error is 1/6.
    strang = Formula(2, 1, (("1/2", "0"),))
    report = verify(strang, include_leading=True)

    assert report.passed
    assert len(report.residuals_by_degree) == 3
    assert float(report.residuals_by_degree[2]) == pytest.approx(1 / 6)


def test_verification_rejects_order_error_in_normalized_positive_formula():
    # Equal B coefficients are normalized and contractive, but the split has order 2.
    formula = Formula(4, 2, (("1/3", "0"), ("1/3", "0")))
    report = verify(formula)

    assert not report.passed
    assert report.min_real_b > 0
    assert report.residuals_by_degree[0] <= report.tolerance
    assert report.residuals_by_degree[2] > report.tolerance


def test_verification_rejects_noncontractive_formula():
    formula = Formula(2, 2, (("-1/10", "0"), ("6/5", "0")))
    report = verify(formula)

    assert not report.passed
    assert report.max_residual <= report.tolerance
    assert report.min_real_b < 0


def test_verification_detects_consistency_error_hidden_by_cancellation():
    large = 2**110
    mismatch = Fraction("1e-62")
    formula = Formula(2, 2, (("1/4", str(large)), ("1/2", str(-2 * large + mismatch))))

    report = verify(formula, dps=80, tolerance="1e-63")

    assert not report.passed
    assert float(report.residuals_by_degree[0]) == pytest.approx(1e-62, abs=0)


def test_verification_increases_precision_for_cancelling_word_terms():
    formula = Formula(2, 2, (("1/3", "1e60"), ("1/3", "-2e60")))

    report = verify(formula, dps=80, tolerance="1e-63")

    assert report.passed
    assert report.residuals_by_degree[0] == 0
    assert report.residuals_by_degree[1] <= report.tolerance
    assert report.dps > 80


def test_verification_accepts_an_arbitrary_precision_tolerance():
    strang = Formula(2, 1, (("1/2", "0"),))
    ctx = mp.clone()
    ctx.dps = 100
    tolerance = ctx.mpf("1e-63")

    report = verify(strang, dps=100, tolerance=tolerance)

    assert report.passed
    assert report.tolerance == tolerance


def test_refinement_improves_archived_order_eight():
    path = Path(__file__).resolve().parents[1] / "coefficients" / "order8_m52.json"
    archived = load_formula(path)
    refined = refine(archived, digits=84)
    tolerance = "1e-82"

    assert not verify(archived, dps=105, tolerance=tolerance).passed
    assert verify(refined, dps=105, tolerance=tolerance).passed
    assert refined.order == archived.order
    assert refined.m == archived.m


def test_refinement_to_fewer_digits_truncates_towards_zero():
    original = Formula(2, 2, (("1.23459e-60", "-0.234569"), ("1/3", "-9/7")))
    truncated = refine(original, digits=3)

    actual = tuple(tuple(Fraction(component) for component in pair) for pair in truncated.b_half)
    assert actual == (
        (Fraction("1.23e-60"), Fraction("-0.234")),
        (Fraction("0.333"), Fraction("-1.28")),
    )
    assert truncated.m == original.m
