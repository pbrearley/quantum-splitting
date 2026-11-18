"""Analytic search derivatives and end-to-end search checks."""

from fractions import Fraction
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from quantum_splitting import Formula, load_formula, search, verify
from quantum_splitting._centered import CenteredConditions
from quantum_splitting._kernel import OrderConditions
from quantum_splitting._words import count_order_conditions, lyndon_words

COEFFICIENTS = Path(__file__).resolve().parents[1] / "coefficients"


def test_order_condition_count_matches_the_solver_words():
    for order in range(2, 13, 2):
        assert count_order_conditions(order) == len(lyndon_words(order))


def test_centered_conditions_match_example_root_and_directional_derivative():
    formula = load_formula(COEFFICIENTS / "order8_m52.json")
    kernel = CenteredConditions(formula.order, formula.m)
    coefficients = import_module("quantum_splitting.search")._complex_coefficients(formula)
    offsets = formula.m * np.cumsum(coefficients[:-1]) - np.arange(formula.m) - 0.5
    residual, jacobian = kernel.evaluate(offsets, derivatives=True)
    assert len(residual) == count_order_conditions(8)
    assert np.linalg.norm(residual / np.linalg.norm(jacobian, axis=1)) < 1e-11

    rng = np.random.default_rng(25)
    direction = rng.normal(size=formula.m) + 1j * rng.normal(size=formula.m)
    step = 1e-6
    difference = (
        kernel.evaluate(offsets + step * direction) - kernel.evaluate(offsets - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(jacobian @ direction, difference, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("m", [6, 7])
def test_search_jacobian_matches_complex_directional_difference(m):
    conditions = OrderConditions(6, m)
    rng = np.random.default_rng(25)
    free = 0.03 * (
        rng.standard_normal(conditions.n_free) + 1j * rng.standard_normal(conditions.n_free)
    )
    direction = rng.standard_normal(free.size) + 1j * rng.standard_normal(free.size)
    direction /= np.linalg.norm(direction)
    step = 1e-6

    residual, jacobian = conditions.residual_jacobian(free)
    difference = (
        conditions.residual(free + step * direction) - conditions.residual(free - step * direction)
    ) / (2 * step)

    np.testing.assert_allclose(residual, conditions.residual(free), atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(jacobian @ direction, difference, atol=1e-8, rtol=2e-7)
    coefficients = conditions.decode(free)
    np.testing.assert_array_equal(coefficients, coefficients[::-1])
    np.testing.assert_allclose(coefficients.sum(), 1)
    np.testing.assert_allclose(conditions.encode(coefficients), free, atol=1e-14, rtol=1e-12)


def test_seeded_search_keeps_example_precision():
    example = load_formula(COEFFICIENTS / "order8_m52.json")
    result = search(8, example.m, initial=example)

    assert result.converged
    assert result.attempts == 0
    assert result.formula == example


def test_seeded_search_checks_the_real_floor_as_written():
    seed = Formula(2, 2, (("0.03", "0"), ("0.94", "0")))

    result = search(2, 2, initial=seed, min_real=0.03, attempts=1, verbose=False)

    assert result.converged
    assert result.attempts == 1
    assert all(Fraction(real) > Fraction("0.03") for real, _ in result.formula.b)


def test_search_checks_the_upper_floor_limit_exactly():
    result = search(2, 2, min_real=0.3333333333333333, attempts=1, verbose=False)

    assert result.converged
    assert all(Fraction(real) == Fraction(1, 3) for real, _ in result.formula.b)
    with pytest.raises(ValueError, match="below 1/\\(m \\+ 1\\)"):
        search(2, 2, min_real=Fraction(1, 3), attempts=1, verbose=False)


def test_seeded_random_search_finds_order_four():
    result = search(order=4, m=4, seed=0, attempts=10)

    assert result.converged
    assert 1 <= result.attempts <= 10
    assert result.formula.order == 4
    assert result.formula.m == 4
    assert verify(result.formula, tolerance="1e-11").passed


def test_tight_search_refines_and_reports_the_candidate():
    improvements = []
    result = search(4, 4, attempts=1, tolerance=1e-70, on_improvement=improvements.append)

    assert result.converged
    assert improvements[-1] == result
    assert verify(result.formula, dps=90, tolerance="1e-70").passed


def test_centered_search_handles_an_odd_number_of_a_factors_with_a_real_floor():
    result = search(4, 5, seed=1, attempts=1, min_real=0.02, tolerance=1e-63)

    assert result.converged
    assert all(Fraction(real) > Fraction("0.02") for real, _ in result.formula.b)
    assert verify(result.formula, dps=84, tolerance="1e-63").passed


def test_progress_callback_counts_unsuccessful_attempts():
    calls = []
    result = search(
        6, 16, attempts=3, max_nfev=1, on_progress=lambda done, best: calls.append((done, best))
    )

    assert not result.converged
    assert [n for n, _ in calls] == [1, 2, 3]
    assert calls[-1][1].formula == result.formula


def test_search_reports_to_stderr_unless_quiet(capsys):
    search(4, 4, attempts=1)
    header = capsys.readouterr().err
    assert "m=4" in header
    assert "2 free complex coefficients, 2 complex order conditions" in header

    search(4, 4, attempts=1, verbose=False)
    assert capsys.readouterr().err == ""


def test_search_rejects_an_undersized_m_before_attempting_a_solve(monkeypatch):
    def unexpected_attempt(*args):
        pytest.fail("The search tried to solve an undersized system")

    monkeypatch.setattr(
        import_module("quantum_splitting.search"), "_attempt", unexpected_attempt
    )

    with pytest.raises(ValueError) as error:
        search(6, 15, attempts=1, verbose=False)

    message = str(error.value)
    assert "7 free complex coefficients" in message
    assert "8 complex order conditions" in message
    assert "m >= 16" in message


def test_screen_passes_an_archived_root():
    search_module = import_module("quantum_splitting.search")
    assert search_module._lyndon_residual(load_formula(COEFFICIENTS / "order6_m16.json")) < 1e-14


def test_screen_bounds_the_verified_residual_from_below():
    search_module = import_module("quantum_splitting.search")
    archived = load_formula(COEFFICIENTS / "order6_m16.json")
    b = search_module._complex_coefficients(archived)
    # Keep the palindrome and consistency while breaking the order conditions.
    b[[1, -2]] += 1e-3
    b[[2, -3]] -= 1e-3
    options = search_module._SearchOptions(6, 16, 0, None, Fraction(0), 1600, 1e-12)
    formula = search_module._formula(options, b)

    estimate = search_module._lyndon_residual(formula)
    verified = verify(formula, dps=40, tolerance="1e-12").max_residual
    assert 1e-10 < estimate <= float(verified) * (1 + 1e-9)


def test_screen_defers_an_inconsistent_formula_to_verification():
    search_module = import_module("quantum_splitting.search")
    archived = load_formula(COEFFICIENTS / "order6_m16.json")
    real, imag = archived.b_half[0]
    shifted = (str(Fraction(real) + Fraction(1, 10**6)), imag)
    formula = Formula(archived.order, archived.m, (shifted, *archived.b_half[1:]))
    assert np.isnan(search_module._lyndon_residual(formula))
