"""Manifold optimisation, full-word preservation and leading error gradients."""

from dataclasses import replace
from fractions import Fraction
from importlib import import_module
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from mpmath import mp

from quantum_splitting import (
    Formula,
    load_formula,
    optimize,
    save_formula,
    verify,
)
from quantum_splitting._kernel import LeadingError, OrderConditions, WordErrors
from quantum_splitting.optimize import OptimizationResult

COEFFICIENTS = Path(__file__).resolve().parents[1] / "coefficients"


@pytest.mark.parametrize("m", [6, 7])
def test_leading_error_gradient_matches_real_directional_derivative(m):
    conditions = OrderConditions(6, m)
    leading_error = LeadingError(6, m)
    rng = np.random.default_rng(93)
    y = 0.03 * (
        rng.standard_normal(conditions.n_free) + 1j * rng.standard_normal(conditions.n_free)
    )
    direction = rng.standard_normal(conditions.n_free) + 1j * rng.standard_normal(conditions.n_free)
    direction /= np.linalg.norm(direction)
    step = 1e-6

    value, gradient = leading_error.value_gradient(y)

    assert value == pytest.approx(leading_error.value(y), rel=1e-12)
    for delta in (direction, 1j * direction):
        finite_difference = (
            leading_error.value(y + step * delta) - leading_error.value(y - step * delta)
        ) / (2 * step)
        assert np.vdot(gradient, delta).real == pytest.approx(
            finite_difference, rel=1e-6, abs=1e-14
        )


@pytest.mark.parametrize("m", [6, 7])
def test_word_error_jacobian_matches_central_differences(m):
    evaluator = WordErrors(4, m)
    rng = np.random.default_rng(17)
    y = 0.1 * (rng.standard_normal(evaluator.n_free) + 1j * rng.standard_normal(evaluator.n_free))
    step = 1e-6

    _, jacobian = evaluator.residual_jacobian(y)

    # The residual is holomorphic in the complex free coordinates.
    for column, basis in enumerate(np.eye(evaluator.n_free)):
        finite_difference = (
            evaluator.residual(y + step * basis) - evaluator.residual(y - step * basis)
        ) / (2 * step)
        np.testing.assert_allclose(jacobian[:, column], finite_difference, rtol=0, atol=1e-8)


def test_leading_error_norm_matches_strang_coefficients():
    strang = Formula(2, 1, (("1/2", "0"),))
    # Among the eight words, ABA has error -1/6, AAB/BAA/BAB have 1/12,
    # ABB/BBA have -1/24, and the pure words have zero error.
    expected = sqrt(5 / 96)
    conditions = OrderConditions(2, 1)
    numerical = sqrt(LeadingError(2, 1).value(np.zeros(conditions.n_free)))

    assert numerical == pytest.approx(expected, rel=1e-13)
    assert float(verify(strang, dps=60, include_leading=True).leading_l2) == pytest.approx(
        expected, rel=1e-13
    )
    assert verify(strang).leading_l2 is None


def _objective_value(formula, objective):
    if objective == "error":
        return float(verify(formula, dps=70, include_leading=True).leading_l2)
    ctx = mp.clone()
    ctx.dps = 70
    b = formula.b_values(ctx)
    if objective == "l1":
        return float(ctx.fsum(abs(z) for z in b))
    return float(ctx.sqrt(ctx.fsum(abs(z) ** 2 for z in b)))


@pytest.mark.parametrize(
    ("objective", "floor"),
    [("l1", 0.006), ("l2", 0.0), ("error", 0.0)],
)
def test_optimising_a_formula_with_free_directions_keeps_all_word_conditions(
    tmp_path, objective, floor
):
    # Two half steps of the order-4 formula, with the adjacent B factors combined.
    seed = Formula(
        4,
        8,
        (
            ("1/20", "-1/60"),
            ("2/15", "1/15"),
            ("2/15", "-1/10"),
            ("2/15", "1/15"),
            ("1/10", "-1/30"),
        ),
    )
    result = optimize(seed, objective=objective, min_real=floor, digits=36, max_iterations=50)
    output = tmp_path / f"{objective}.json"

    save_formula(result.formula, output)
    saved = load_formula(output)

    assert result.improved
    assert result.objective == objective
    assert saved == result.formula
    assert (saved.order, saved.m) == (seed.order, seed.m)
    assert all(Fraction(real) > Fraction(str(floor)) for real, _ in saved.b)
    report = verify(saved, dps=60, tolerance="1e-35")
    assert report.passed
    assert result.residual <= report.tolerance
    initial = _objective_value(seed, objective)
    final = _objective_value(saved, objective)
    assert result.initial_value == pytest.approx(initial, rel=1e-9, abs=0)
    assert result.value == pytest.approx(final, rel=1e-9, abs=0)
    assert final < initial * 0.999


def test_order_two_l2_optimisation_reaches_the_uniform_formula():
    seed = Formula(2, 3, (("1/10", "1/20"), ("2/5", "-1/20")))

    result = optimize(seed, objective="l2", digits=30)

    assert result.improved
    assert result.converged
    assert result.value == pytest.approx(0.5, abs=1e-10)
    assert verify(result.formula, dps=55, tolerance="1e-29").passed


def test_optimisation_without_a_budget_restarts_from_the_verified_best(monkeypatch):
    seed = Formula(2, 3, (("1/10", "1/20"), ("2/5", "-1/20")))
    best = Formula(2, 3, (("1/4", "0"), ("1/4", "0")))
    improved = OptimizationResult(
        formula=best,
        objective="l2",
        initial_value=0.6,
        value=0.5,
        residual=mp.mpf(0),
        iterations=300,
        converged=False,
        improved=True,
        message="Iteration limit reached",
    )
    stalled = replace(improved, initial_value=0.5, iterations=12, improved=False)
    calls = []

    def optimise_pass(formula, objective, min_real, digits, max_iterations):
        calls.append(formula)
        return improved if len(calls) == 1 else stalled

    module = import_module("quantum_splitting.optimize")
    monkeypatch.setattr(module, "_optimize_pass", optimise_pass)

    result = optimize(seed, objective="l2")

    assert calls == [seed, best]
    assert result.formula == best
    assert result.initial_value == improved.initial_value
    assert result.iterations == 312
    assert result.improved
    assert not result.converged
    assert "No further verified improvement" in result.message


def test_isolated_example_order_four_has_no_optimisation_step():
    seed = load_formula(COEFFICIENTS / "order4_m4.json")

    result = optimize(seed)

    assert not result.improved
    assert result.iterations == 0
    assert result.formula == seed
    assert result.value == result.initial_value
    assert verify(result.formula, dps=75, tolerance="1e-49").passed


def test_optimisation_rejects_seed_that_fails_order_conditions():
    seed = Formula(4, 4, (("1/5", "0"),) * 3)

    with pytest.raises(ValueError):
        optimize(seed)


@pytest.mark.parametrize("floor", [0.03, 0.031])
def test_optimisation_rejects_seed_at_or_below_its_floor(floor):
    seed = Formula(2, 2, (("0.03", "0"), ("0.94", "0")))

    with pytest.raises(ValueError, match="real-part floor"):
        optimize(seed, min_real=floor)


def test_optimisation_checks_the_upper_floor_limit_exactly():
    seed = Formula(2, 2, (("1/3", "0"),) * 2)

    result = optimize(seed, min_real=0.3333333333333333)

    assert result.formula == seed
    with pytest.raises(ValueError, match="below 1/\\(m \\+ 1\\)"):
        optimize(seed, min_real=Fraction(1, 3))


def test_optimisation_preserves_a_residual_below_float_precision():
    seed = load_formula(COEFFICIENTS / "order4_m4.json")
    result = optimize(seed, digits=400)
    report = verify(result.formula, dps=420, tolerance="1e-399")

    assert report.passed
    assert 0 < result.residual < mp.mpf("1e-399")
    assert result.residual == report.max_residual
