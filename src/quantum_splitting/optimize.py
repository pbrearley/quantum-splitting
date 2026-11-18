"""Local constrained optimisation of a verified product formula."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import factorial
from typing import Literal

import numpy as np
from mpmath import mpf
from numpy.typing import NDArray

from ._bounds import above_real_floor, real_floor
from ._kernel import LeadingError
from ._words import lyndon_words, word_index
from .model import Formula, _mirror
from .precision import (
    RefinementError,
    VerificationResult,
    _consistent_half,
    _context,
    _jacobian,
    _product_series,
    _refine,
    verify,
)

Objective = Literal["l1", "l2", "error"]


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of a local constrained optimisation.

    The initial and final values are evaluated on the input and returned
    coefficient strings. The residual is the maximum absolute word error
    through the requested order, retained as an arbitrary-precision value.
    The improved flag reports a meaningful reduction in the objective.
    Convergence also requires termination at a verified local solver result.
    """

    formula: Formula
    objective: Objective
    initial_value: float
    value: float
    residual: mpf
    iterations: int
    converged: bool
    improved: bool
    message: str


class OptimizationError(RuntimeError):
    """The optimiser could not prepare a verified feasible formula."""


class _Coordinates:
    """Scale displacements from the seed while eliminating the last B value."""

    def __init__(self, formula: Formula) -> None:
        self.formula = formula
        self.ctx = _context(40)
        self.free = formula.b_half_values(self.ctx)[:-1]
        self.size = len(self.free)
        self.scale = 1.0 / (formula.m + 1)
        self.anchor = np.array(
            [complex(v) for v in _consistent_half(self.ctx, self.free, formula.m)]
        )
        last_weight = 1 + formula.m % 2
        self.affine = self.scale * np.vstack(
            (np.eye(self.size), np.full(self.size, -2 / last_weight))
        )
        self.weights = np.full(self.size + 1, 2)
        self.weights[-1] = last_weight

    def half(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        return self.anchor + self.affine @ self.delta(x)

    def exact_half(self, x: NDArray[np.float64]):
        free = tuple(
            b + self.scale * self.ctx.mpc(complex(step))
            for b, step in zip(self.free, self.delta(x))
        )
        return _consistent_half(self.ctx, free, self.formula.m)

    def record(self, x: NDArray[np.float64]) -> Formula:
        pairs = tuple(
            (self.ctx.nstr(z.real, 32), self.ctx.nstr(z.imag, 32)) for z in self.exact_half(x)
        )
        return Formula(self.formula.order, self.formula.m, pairs)

    def delta(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        return x[: self.size] + 1j * x[self.size :]


class _OrderConstraints:
    """Evaluate the order equations at guard precision and whiten their rows."""

    def __init__(self, coordinates: _Coordinates) -> None:
        self.coordinates = coordinates
        self.ctx = coordinates.ctx
        self.words = lyndon_words(coordinates.formula.order)
        self.degree = max(map(len, self.words), default=0)
        self.rows = [(word_index(word), self.ctx.one / factorial(len(word))) for word in self.words]
        self._point = None
        self._series = None
        self._raw_jacobian = None

    def prepare(self, start: NDArray[np.float64]) -> int:
        if not self.words or not self.coordinates.size:
            return 0
        jacobian = self._jacobian(start)
        norms = np.linalg.norm(jacobian, axis=1)
        norms[norms == 0] = 1
        left, singular, _ = np.linalg.svd(jacobian / norms[:, None], full_matrices=False)
        threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0]
        rank = int(np.count_nonzero(singular > threshold))
        if rank == len(self.words):
            self.whitening = left.conj().T / singular[:, None] / norms[None, :]
            self.precise_whitening = self.ctx.matrix(self.whitening.tolist())
        return rank

    def _expand(self, x: NDArray[np.float64]):
        if self._point is None or not np.array_equal(x, self._point):
            m = self.coordinates.formula.m
            b = _mirror(self.coordinates.exact_half(x), m)
            self._series = _product_series(self.ctx, self.degree, m, b, keep_prefixes=True)
            self._point = x.copy()
            self._raw_jacobian = None
        return self._series

    def _jacobian(self, x: NDArray[np.float64]) -> NDArray[np.complex128]:
        _, before, after = self._expand(x)
        if self._raw_jacobian is None:
            jacobian = _jacobian(self.ctx, self.words, before, after, self.coordinates.formula.m)
            self._raw_jacobian = self.coordinates.scale * np.asarray(
                jacobian.tolist(), dtype=np.complex128
            )
        return self._raw_jacobian

    def residual(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        series, _, _ = self._expand(x)
        raw = self.ctx.matrix([series[index] - target for index, target in self.rows])
        return _stack(np.asarray(list(self.precise_whitening * raw), dtype=np.complex128))

    def jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        jacobian = self.whitening @ self._jacobian(x)
        return np.block([[jacobian.real, -jacobian.imag], [jacobian.imag, jacobian.real]])


def _stack(values: NDArray[np.complex128]) -> NDArray[np.float64]:
    return np.concatenate((values.real, values.imag))


def _objective_evaluator(objective: Objective, coordinates: _Coordinates):
    if objective == "error":
        formula = coordinates.formula
        leading = LeadingError(formula.order, formula.m)
        anchor = leading.basis.T @ (coordinates.anchor - leading.reference)
        transform = leading.basis.T @ coordinates.affine

        def evaluate(x):
            value, gradient = leading.value_gradient(anchor + transform @ coordinates.delta(x))
            return value, _stack(transform.T @ gradient)

        return evaluate

    def evaluate(x):
        half = coordinates.half(x)
        magnitude = np.abs(half)
        if objective == "l1":
            value = coordinates.weights @ magnitude
            direction = np.divide(half, magnitude, out=np.zeros_like(half), where=magnitude != 0)
        else:
            value = coordinates.weights @ (magnitude**2)
            direction = 2 * half
        gradient = coordinates.affine.T @ (coordinates.weights * direction)
        return float(value), _stack(gradient)

    return evaluate


def _stored_value(formula: Formula, objective: Objective, report: VerificationResult):
    if objective == "error":
        assert report.leading_l2 is not None
        return report.leading_l2
    ctx = _context(report.dps)
    magnitudes = [abs(value) for value in formula.b_values(ctx)]
    if objective == "l1":
        return ctx.fsum(magnitudes)
    return ctx.sqrt(ctx.fsum(value**2 for value in magnitudes))


def optimize(
    formula: Formula,
    *,
    objective: Objective = "l1",
    min_real: float | str = 0.0,
    digits: int = 50,
    max_iterations: int | None = None,
) -> OptimizationResult:
    """Minimise a coefficient or leading word error norm from a feasible seed.

    The seed must pass full-word verification at a tolerance of 1e-12.
    The stored real B coefficients must strictly exceed the exact decimal
    value of ``min_real``.
    The returned coefficients are verified at a tolerance of 10**(1-digits).
    The objective is evaluated locally in double precision, while the order
    constraints and final verification use arbitrary precision. By default
    the optimiser periodically recomputes the constraint scaling around the
    best verified iterate. It continues until the local solve converges or
    a pass finds no further verified improvement. Supply max_iterations to
    run a single local solve with that budget.

    The ``error`` objective is the Euclidean norm of all degree
    formula.order + 1 word coefficient errors. The l1 and l2 objectives use
    the full sequence of B coefficients.
    """
    if max_iterations is not None:
        return _optimize_pass(formula, objective, min_real, digits, max_iterations)

    # Whitening at a fixed anchor becomes inaccurate over long displacements.
    # Refresh it at intervals, stopping on the result rather than on a total budget.
    reanchor_interval = 300
    result = _optimize_pass(formula, objective, min_real, digits, reanchor_interval)
    initial_value = result.initial_value
    iterations = result.iterations
    improved = result.improved
    while result.improved and not result.converged:
        result = _optimize_pass(result.formula, objective, min_real, digits, reanchor_interval)
        iterations += result.iterations
    message = result.message
    if not result.converged and result.iterations:
        message = "No further verified improvement at the solver tolerance"
    return replace(
        result,
        initial_value=initial_value,
        iterations=iterations,
        improved=improved,
        message=message,
    )


def _optimize_pass(
    formula: Formula,
    objective: Objective,
    min_real: float | str,
    digits: int,
    max_iterations: int,
) -> OptimizationResult:
    if objective not in ("l1", "l2", "error"):
        raise ValueError("objective must be 'l1', 'l2', or 'error'")
    if formula.order < 2 or formula.order % 2:
        raise ValueError("optimisation requires an even order of at least 2")
    floor = real_floor(min_real, formula.m)
    if type(digits) is not int or digits < 15:
        raise ValueError("digits must be an integer of at least 15")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    dps = max(50, digits + 20)
    leading = objective == "error"

    seed_report = verify(formula, dps=dps, tolerance="1e-12", include_leading=leading)
    if not seed_report.passed or not above_real_floor(formula, floor):
        raise ValueError("seed must satisfy the order conditions at 1e-12 and the real-part floor")
    initial_value = _stored_value(formula, objective, seed_report)
    target = f"1e{1 - digits}"

    def checked(candidate: Formula):
        if not above_real_floor(candidate, floor):
            return None
        report = verify(candidate, dps=dps, tolerance=target, include_leading=leading)
        if not report.passed:
            try:
                candidate = _refine(candidate, digits=digits, max_iterations=30)
            except RefinementError:
                return None
            if not above_real_floor(candidate, floor):
                return None
            report = verify(candidate, dps=dps, tolerance=target, include_leading=leading)
        if not report.passed:
            return None
        return candidate, _stored_value(candidate, objective, report), report.max_residual

    baseline = checked(formula)
    if baseline is None:
        raise OptimizationError("could not verify the seed at the requested output precision")

    # Changes at this relative scale are below the stopping criterion of the solve.
    meaningful = 1e-11

    def outcome(candidate, iterations, converged, message):
        chosen, value, residual = candidate
        return OptimizationResult(
            chosen,
            objective,
            float(initial_value),
            float(value),
            residual,
            iterations,
            converged,
            value < initial_value * (1.0 - meaningful),
            message,
        )

    coordinates = _Coordinates(baseline[0])
    start = np.zeros(2 * coordinates.size)
    equations = _OrderConstraints(coordinates)
    rank = equations.prepare(start)
    if rank == coordinates.size:
        return outcome(baseline, 0, True, "No free local directions at the seed")
    if rank < len(equations.words):
        return outcome(baseline, 0, False, "Order constraints are rank deficient at the seed")

    evaluate = _objective_evaluator(objective, coordinates)
    objective_scale, _ = evaluate(start)
    degree = formula.order + 1
    numerical_floor = (np.finfo(float).eps * (2**degree) / factorial(degree)) ** 2
    if leading and objective_scale <= numerical_floor:
        return outcome(baseline, 0, True, "Leading error is at the optimisation precision floor")

    from scipy.optimize import minimize

    def scaled_objective(x):
        value, gradient = evaluate(x)
        return value / objective_scale, gradient / objective_scale

    # Leave a small buffer at the inequality boundary for final retraction.
    anchor_min = np.min(coordinates.anchor.real)
    numerical_bound = float(floor)
    margin = anchor_min - numerical_bound
    working_floor = min(anchor_min, numerical_bound + min(1e-8, 0.5 * margin))

    def positive(x):
        return coordinates.half(x).real - working_floor

    positive_jacobian = np.hstack((coordinates.affine, np.zeros_like(coordinates.affine)))
    constraints = [{"type": "ineq", "fun": positive, "jac": lambda _: positive_jacobian}]
    if equations.words:
        constraints.append({"type": "eq", "fun": equations.residual, "jac": equations.jacobian})

    # Keep promising iterates in case the final point can't be refined.
    candidates = []

    def remember(x):
        if not np.all(np.isfinite(x)) or np.min(positive(x)) < -1e-9:
            return
        if equations.words and np.max(np.abs(equations.residual(x))) > 1e-4:
            return
        merit, _ = scaled_objective(x)
        if merit < 1.0 - meaningful:
            candidates.append((merit, x.copy()))

    solution = minimize(
        scaled_objective,
        start,
        jac=True,
        constraints=constraints,
        method="SLSQP",
        callback=remember,
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    terminal = checked(coordinates.record(solution.x)) if np.all(np.isfinite(solution.x)) else None
    best = baseline
    if terminal is not None and terminal[1] < baseline[1] * (1.0 - meaningful):
        best = terminal
    for merit, point in sorted(candidates, key=lambda item: item[0]):
        # Only restore intermediates whose provisional merit could still
        # improve on the verified incumbent.
        best_value = float(best[1])
        best_merit = best_value if objective == "l1" else best_value**2
        if merit >= best_merit / objective_scale * (1.0 - meaningful):
            break
        if np.array_equal(point, solution.x):
            continue
        fallback = checked(coordinates.record(point))
        if fallback is not None and fallback[1] < best[1] * (1.0 - meaningful):
            best = fallback
    if best is terminal:
        return outcome(best, solution.nit, bool(solution.success), str(solution.message))
    if best is not baseline:
        return outcome(
            best,
            solution.nit,
            False,
            f"Returned a verified intermediate iterate ({solution.message})",
        )
    converged = (
        bool(solution.success)
        and terminal is not None
        and terminal[1] <= baseline[1] * (1.0 + meaningful)
    )
    message = (
        str(solution.message)
        if converged
        else f"No verified improvement was found ({solution.message})"
    )
    return outcome(baseline, solution.nit, converged, message)
