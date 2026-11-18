"""Free coordinates and a damped least-squares solver for centered conditions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import lstsq, svd

from ._centered import CenteredConditions


def _bounded_step(delta, right_h, inverse, bounds, lower):
    """Correct a step in its quadratic metric to meet the real inequalities."""
    original = delta
    active = []
    for _ in range(28):
        violation = lower - bounds @ delta.real
        if active:
            violation[active] = 0.0
        index = int(np.argmax(violation))
        if violation[index] <= 1e-14:
            break
        active.append(index)
        for _ in range(12):
            constraints = bounds[active]
            # Columns of H⁻¹ Aᵀ map the active multipliers to a step correction.
            directions = right_h.conj().T @ (inverse[:, None] * (right_h @ constraints.T))
            gram = (constraints @ directions).real
            rhs = constraints @ original.real - lower[active]
            multipliers = lstsq(gram, rhs, cond=1e-12, check_finite=False)[0]
            drop = int(np.argmax(multipliers))
            if multipliers[drop] <= 1e-14:
                delta = original - directions @ multipliers
                break
            active.pop(drop)
            if not active:
                return original
    return delta


class SearchConditions:
    """Represent consistent palindromes with the linear moments eliminated."""

    def __init__(self, order: int, m: int) -> None:
        self.m = m
        self._half_intervals = m // 2
        self._conditions = CenteredConditions(order, m)
        identity = np.eye(self._half_intervals)
        mirror = np.vstack((identity, np.zeros((m % 2, self._half_intervals)), -identity[::-1]))

        linear = np.array(
            [
                len(word) == 1 and self._conditions.alphabet[word[0]][0] == 1
                for word in self._conditions.words
            ]
        )
        self._nonlinear = ~linear
        residual, jacobian = self._conditions.evaluate(np.zeros(m, dtype=complex), True)
        moments = (jacobian.real @ mirror)[linear]
        scales = np.linalg.norm(moments, axis=1)
        moments /= scales[:, None]
        center = np.linalg.lstsq(moments, -residual[linear].real / scales, rcond=None)[0]
        _, _, right_h = np.linalg.svd(moments, full_matrices=True)
        self._basis = right_h[np.count_nonzero(linear) :].T
        self.n_free = self._basis.shape[1]
        self._center = center
        self._offset = mirror @ center
        self._q_basis = mirror @ self._basis

        # For an even m, the middle B coefficient spans two mirrored offsets.
        # For an odd m, it spans one offset and the central interval at q=0.
        middle = 2 - m % 2
        self._b_offset = np.r_[center[0] + 0.5, 1 + np.diff(center), 1 - middle * center[-1]] / m
        self._b_basis = (
            np.vstack((self._basis[0], np.diff(self._basis, axis=0), -middle * self._basis[-1])) / m
        )

        degree = self._conditions.degrees[self._nonlinear]
        lengths = np.array([len(word) for word in self._conditions.words])[self._nonlinear]
        self._stages = tuple(
            rows
            for rows in ((degree <= 5) | (lengths == 1), (degree <= 7) | (lengths <= 2))
            if not np.all(rows)
        )

    def encode(self, b: ArrayLike) -> NDArray[np.complex128]:
        """Project a coefficient set into the affine centered coordinates."""
        n = self._half_intervals
        half = self.m * np.cumsum(np.asarray(b)[:n]) - np.arange(n) - 0.5
        return self._basis.T @ (half - self._center)

    def decode(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Recover all m+1 B coefficients, including their mirrored half."""
        offsets = self._center + self._basis @ x
        middle = 2 - self.m % 2
        half = np.hstack((offsets[0] + 0.5, 1 + np.diff(offsets), 1 - middle * offsets[-1]))
        half /= self.m
        return np.concatenate((half, half[: (self.m + 1) // 2][::-1]))

    def partial(self, x: NDArray[np.complex128], derivatives: bool = False):
        """Evaluate the remaining equations and, optionally, their Jacobian."""
        q = self._offset + self._q_basis @ x
        if derivatives:
            residual, jacobian = self._conditions.evaluate(q, True)
            return residual[self._nonlinear], jacobian[self._nonlinear] @ self._q_basis
        return self._conditions.evaluate(q)[self._nonlinear]

    def start(self, seed: int, index: int, min_real: float) -> NDArray[np.complex128] | None:
        """Draw a feasible start with Gaussian real and logistic imaginary offsets."""
        n = self._half_intervals
        rng = np.random.default_rng(np.random.SeedSequence((seed, index)))
        width = (0.03, 0.15, 0.30, 0.55)[(seed + index) % 4]
        real = width * rng.normal(size=n)
        imag = rng.logistic(scale=1 / (2 * np.pi), size=n)

        interior = np.zeros(self.n_free)
        margin = self._b_offset - min_real
        if np.min(margin) <= 0:
            interior = _bounded_step(
                interior, np.eye(self.n_free), np.ones(self.n_free), self._b_basis, 1e-12 - margin
            )
            margin += self._b_basis @ interior
            if np.min(margin) <= 0:
                return None

        # Match the real and imaginary parts of the unweighted second moment.
        # With one complex degree of freedom the real part is fixed instead.
        xr = self._basis.T @ real if self.n_free > 1 else np.zeros(self.n_free)
        movement = self._b_basis @ xr
        closing = movement < 0
        if np.any(closing):
            # Preserve the historical scale when its trial is feasible.
            ratios = -movement[closing]
            scale = min(1.0, 0.9 * float(np.min(self._b_offset[closing] / ratios)))
            if np.any(scale * ratios >= margin[closing]):
                scale = min(scale, 0.9 * np.min(margin[closing] / ratios))
            xr *= scale
        xr += interior
        qreal = self._center + self._basis @ xr
        xi = self._basis.T @ imag
        variance = np.dot(xr, xr)
        if variance > 1e-20:
            xi -= (np.dot(xr, xi) / variance) * xr
        if np.linalg.norm(xi) < 1e-12:
            xi = np.zeros(self.n_free)
            xi[np.argmin(np.abs(xr))] = 1.0
            if variance > 1e-20 and self.n_free > 1:
                xi -= (np.dot(xr, xi) / variance) * xr
        xi *= np.sqrt(self.m / 24 + np.dot(qreal, qreal)) / np.linalg.norm(xi)
        return xr + 1j * xi

    def solve(self, x, max_nfev: int, min_real: float):
        """Fit successively larger subsets of conditions within one budget."""
        used = 0
        for rows in self._stages:
            budget = min(400, max_nfev // 4, max_nfev - used - 1)
            if budget > 0:
                x, _, count = self._solve_rows(x, rows, budget, 170, min_real)
                used += count
        x, score, count = self._solve_rows(x, slice(None), max_nfev - used, 550, min_real)
        return x, score, used + count

    def _solve_rows(self, x, rows, budget, iteration_limit, min_real):
        damping = 1e-7
        previous = score = float("inf")
        evaluations = 0
        for _ in range(iteration_limit):
            if evaluations >= budget:
                break
            residual, jacobian = self.partial(x, True)
            evaluations += 1
            residual, jacobian = residual[rows], jacobian[rows]
            norms = np.maximum(np.linalg.norm(jacobian, axis=1), 1e-150)
            residual = residual / norms
            jacobian = jacobian / norms[:, None]
            score = float(np.linalg.norm(residual))
            if score < 1e-12 or not np.isfinite(score):
                break
            try:
                left, singular, right_h = svd(jacobian, full_matrices=True, check_finite=False)
            except np.linalg.LinAlgError:
                break
            if singular[0] == 0:
                break
            # Give weak directions some weight while limiting amplification
            # at a numerically singular iterate.
            scale = np.maximum(singular / singular[0], 3e-7) ** -0.30
            weighted = singular * scale
            projected = (left.conj().T @ residual) * scale
            merit = np.linalg.norm(projected)
            margin = self._b_offset + self._b_basis @ x.real - min_real
            if margin.min() <= 0:
                break
            accepted = False
            for _ in range(8):
                if evaluations >= budget:
                    break
                inverse = np.full(right_h.shape[0], 1 / damping)
                modes = inverse[: singular.size]
                modes[:] = 1 / (weighted * weighted + damping)
                delta = right_h[: singular.size].conj().T @ (-weighted * modes * projected)
                movement = self._b_basis @ delta.real
                if np.any(movement < -0.5 * margin):
                    delta = _bounded_step(delta, right_h, inverse, self._b_basis, -0.99 * margin)
                    movement = self._b_basis @ delta.real
                closing = movement < 0
                step = min(1.0, 1.0 / max(1e-14, np.max(np.abs(self._q_basis @ delta))))
                if np.any(closing):
                    step = min(step, 0.999 * np.min(margin[closing] / -movement[closing]))
                for _ in range(8):
                    if evaluations >= budget or step < 1e-12:
                        break
                    trial = x + step * delta
                    trial_residual = self.partial(trial)[rows] / norms
                    evaluations += 1
                    trial_merit = np.linalg.norm((left.conj().T @ trial_residual) * scale)
                    if (
                        trial_merit < merit
                        and np.min(self._b_offset + self._b_basis @ trial.real) > min_real
                    ):
                        x = trial
                        trial_score = float(np.linalg.norm(trial_residual))
                        accepted = True
                        damping = max(1e-27, damping * (0.23 if step > 0.5 else 0.8))
                        break
                    step *= 0.5
                if accepted:
                    break
                damping *= 10
            if not accepted:
                break
            if abs(previous - score) <= 1e-12 * score:
                score = trial_score
                break
            previous, score = score, trial_score
        return x, score, evaluations
