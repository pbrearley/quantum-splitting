"""Arbitrary-precision verification and refinement of symmetric product formulas."""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from math import factorial

from mpmath import mp
from mpmath.ctx_iv import MPIntervalContext
from mpmath.libmp import to_rational

from ._words import append_terms, derivative_splits, lyndon_words, word_index
from .model import Formula, _mirror


@dataclass(frozen=True)
class VerificationResult:
    """Maximum absolute errors in the associative word coefficients.

    ``residuals_by_degree[d - 1]`` is the maximum over all words of length
    ``d``. ``max_residual`` covers degrees 1 through ``order``. ``passed``
    also requires that every B coefficient have positive real part. If
    requested, ``leading_l2`` is the Euclidean norm of all degree ``order + 1``
    word errors.
    """

    order: int
    dps: int
    tolerance: mp.mpf
    max_residual: mp.mpf
    residuals_by_degree: tuple[mp.mpf, ...]
    min_real_b: mp.mpf
    passed: bool
    leading_l2: mp.mpf | None = None


class RefinementError(RuntimeError):
    """A seed could not be refined to the requested accuracy."""


def _context(dps):
    ctx = mp.clone()
    ctx.dps = dps
    return ctx


def _append_exponential(series, order, letter, coefficient):
    """Multiply a truncated word series on the right by exp(coefficient*letter)."""
    powers = [1]
    for k in range(1, order + 1):
        powers.append(powers[-1] * coefficient / k)
    # Every source word is shorter than its destination. Descending index
    # order preserves the old source coefficients during the in-place update.
    for dest, sources in append_terms(order, letter):
        for source, power in sources:
            series[dest] += series[source] * powers[power]


def _product_series(ctx, order, m, b, *, keep_prefixes=False):
    """Expand the product and optionally retain prefixes around each B factor."""
    series = [ctx.zero] * (1 << (order + 1))
    series[1] = ctx.one  # The empty word has index 1.
    a = ctx.one / m
    before_b, after_b = [], []
    for index, coefficient in enumerate(b):
        if index:
            _append_exponential(series, order, "A", a)
        if keep_prefixes:
            before_b.append(series.copy())
        _append_exponential(series, order, "B", coefficient)
        if keep_prefixes:
            after_b.append(series.copy())
    return series, before_b, after_b


def _fraction_value(ctx, value):
    return ctx.mpf(value.numerator) / value.denominator


def _exact_components(formula):
    real, imag = zip(*((Fraction(re), Fraction(im)) for re, im in formula.b))
    consistency_squared = (sum(real) - 1) ** 2 + sum(imag) ** 2
    magnitude = 1 + sum(map(abs, real)) + sum(map(abs, imag))
    return min(real), consistency_squared, magnitude


def _guarded_dps(ctx, order, m, magnitude, tolerance):
    # Terms at degree d are bounded by the dth power of the absolute
    # coefficient sum. Reserve digits for cancellation at that scale.
    scale = order * ctx.log10(_fraction_value(ctx, magnitude))
    guard = len(str((2 * m + 1) * order)) + 8
    return max(ctx.dps, int(ctx.ceil(scale - ctx.log10(tolerance))) + guard)


def _word_errors(ctx, formula, order, consistency_squared):
    series, _, _ = _product_series(ctx, order, formula.m, formula.b_values(ctx))
    errors = [ctx.sqrt(_fraction_value(ctx, consistency_squared))]
    leading_l2 = None
    for degree in range(2, order + 1):
        target = ctx.one / factorial(degree)
        start, stop = 1 << degree, 1 << (degree + 1)
        word_errors = [abs(value - target) for value in series[start:stop]]
        errors.append(max(word_errors))
        if degree > formula.order:
            leading_l2 = ctx.sqrt(ctx.fsum(error**2 for error in word_errors))
    return errors, leading_l2


def _word_diagnostics(formula, dps):
    """Measure the word errors and leading norm without certification."""
    _require_positive_int(dps, "working precision")
    ctx = _context(dps)
    _, consistency_squared, magnitude = _exact_components(formula)
    order = formula.order + 1
    ctx.dps = _guarded_dps(ctx, order, formula.m, magnitude, ctx.mpf("1e-12"))
    errors, leading_l2 = _word_errors(ctx, formula, order, consistency_squared)
    return ctx, errors, leading_l2


def _word_verdict(formula, dps, tolerance):
    """Enclose the higher-degree errors with outward-rounded arithmetic."""
    ctx = MPIntervalContext()
    ctx.dps = dps
    limit = _fraction_value(ctx, tolerance)
    order = formula.order
    series, _, _ = _product_series(ctx, order, formula.m, formula.b_values(ctx))
    inconclusive = False
    for degree in range(2, order + 1):
        target = ctx.one / factorial(degree)
        for value in series[1 << degree : 1 << (degree + 1)]:
            error = abs(value - target)
            if error.a > limit.b:
                return False
            if error.b > limit.a:
                inconclusive = True
    return None if inconclusive else True


def verify(
    formula: Formula,
    *,
    dps: int = 80,
    tolerance: str = "1e-12",
    include_leading: bool = False,
) -> VerificationResult:
    """Check every word against ``exp(h*(A+B))`` through ``formula.order``.

    The supplied coefficient strings are evaluated as written, including their
    consistency error after rounding or truncation. ``include_leading`` also
    reports the maximum and Euclidean norm of the degree ``order + 1`` word
    errors, outside the pass criterion.
    ``dps`` sets the minimum decimal working precision, and ``tolerance`` sets
    an absolute bound for each word. Consistency and positivity are checked
    exactly. A passing higher-degree check requires interval bounds within the
    tolerance. Pass a decimal string to specify a tolerance exactly.
    """
    _require_positive_int(dps, "working precision")
    ctx = _context(dps)
    try:
        if hasattr(tolerance, "_mpf_") and ctx.isfinite(tolerance):
            tolerance_exact = Fraction(*to_rational(tolerance._mpf_))
        else:
            tolerance_exact = Fraction(tolerance)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        raise ValueError("tolerance must be finite and positive") from None
    if tolerance_exact <= 0:
        raise ValueError("tolerance must be finite and positive")
    tolerance_value = _fraction_value(ctx, tolerance_exact)
    if tolerance_value <= 10 * ctx.eps:
        raise ValueError("increase working precision for the requested tolerance")

    min_real, consistency_squared, magnitude = _exact_components(formula)
    admissible = min_real > 0 and consistency_squared <= tolerance_exact**2

    order = formula.order + int(include_leading)
    dps = _guarded_dps(ctx, order, formula.m, magnitude, tolerance_value)

    passed = False
    for extra in (0, 30):
        ctx = _context(dps + extra)
        tolerance_value = _fraction_value(ctx, tolerance_exact)
        errors, leading_l2 = _word_errors(ctx, formula, order, consistency_squared)
        max_residual = max(errors[: formula.order])
        if not admissible:
            break
        if formula.order == 1:
            passed = True
            break
        # Screen clear failures before repeating the expansion with intervals.
        if max_residual > 2 * tolerance_value:
            break
        verdict = _word_verdict(formula, ctx.dps, tolerance_exact)
        if verdict is not None:
            passed = verdict
            break

    return VerificationResult(
        order=formula.order,
        dps=ctx.dps,
        tolerance=tolerance_value,
        max_residual=max_residual,
        residuals_by_degree=tuple(errors),
        min_real_b=_fraction_value(ctx, min_real),
        passed=passed,
        leading_l2=leading_l2,
    )


def _consistent_half(ctx, free, m):
    # Every free B coefficient occurs twice. The last occurs once when the
    # product has a central B factor, and twice when it has a central A factor.
    last_weight = 1 + m % 2
    return (*free, (ctx.one - 2 * ctx.fsum(free)) / last_weight)


def _jacobian(ctx, words, before_b, after_b, m):
    """Differentiate selected words in the free, consistency-preserving coordinates."""
    half_size = m // 2 + 1
    last_weight = 1 + m % 2
    rows = []
    for word in words:
        splits = derivative_splits(word)
        half_gradient = [ctx.zero] * half_size
        for j in range(m + 1):
            # d(exp(b*B))/db = exp(b*B)*B. For a palindrome, the suffix
            # after B_j is the reversal of the prefix before B_(m-j).
            left, reversed_right = after_b[j], before_b[m - j]
            derivative = ctx.fsum(left[u] * reversed_right[v] for u, v in splits)
            half_gradient[min(j, m - j)] += derivative
        consistency = (2 // last_weight) * half_gradient[-1]
        rows.append([value - consistency for value in half_gradient[:-1]])
    return ctx.matrix(rows)


def _newton_step(ctx, jacobian, residual):
    """Compute a minimum-norm correction, including for rectangular systems."""
    rhs = -ctx.matrix(residual)
    try:
        if jacobian.rows == jacobian.cols:
            return ctx.lu_solve(jacobian, rhs)
        if jacobian.rows < jacobian.cols:
            adjoint = jacobian.H
            return adjoint * ctx.lu_solve(jacobian * adjoint, rhs)
    except (ZeroDivisionError, ValueError):
        pass
    # The SVD also handles overdetermined and rank-deficient Jacobians.
    left, singular, right = ctx.svd(jacobian)
    cutoff = ctx.sqrt(ctx.eps) * singular[0]
    projected = left.H * rhs
    correction = [
        value / scale if scale > cutoff else ctx.zero for value, scale in zip(projected, singular)
    ]
    return right.H * ctx.matrix(correction)


def _row_weights(ctx, jacobian):
    """Balance each residual by the sensitivity of its order condition."""
    norms = [
        ctx.sqrt(ctx.fsum(abs(jacobian[i, j]) ** 2 for j in range(jacobian.cols)))
        for i in range(jacobian.rows)
    ]
    floor = max(norms) * ctx.sqrt(ctx.eps)
    return [ctx.one / (max(norm, floor) or ctx.one) for norm in norms]


def _require_positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def refine(formula: Formula, *, digits: int = 80, max_iterations: int = 12) -> Formula:
    """Change the coefficient precision to ``digits`` significant figures.

    A target below the stored decimal precision truncates each component
    towards zero. This can change the consistency sum and word residuals.
    Higher targets use Newton refinement and verify the rounded output with
    absolute tolerance ``10**(1 - digits)``. Exact rational input that already
    meets the refinement target is kept exact.

    Raises ``RefinementError`` if Newton refinement cannot reach the target.
    """
    _require_positive_int(digits, "digits")
    _require_positive_int(max_iterations, "max_iterations")
    decimal_precision = max(
        (
            len(decimal.as_tuple().digits)
            for pair in formula.b_half
            for part in pair
            if "/" not in part and (decimal := Decimal(part))
        ),
        default=0,
    )
    if digits < decimal_precision:
        return _truncate(formula, digits=digits)
    return _refine(formula, digits=digits, max_iterations=max_iterations)


def _refine(formula: Formula, *, digits: int = 80, max_iterations: int = 12) -> Formula:
    """Enforce the order conditions and round the refined coefficients."""
    _require_positive_int(digits, "digits")
    _require_positive_int(max_iterations, "max_iterations")
    dps = digits + 30
    ctx = _context(dps)
    target_error = f"1e-{digits + 10}"
    tolerance = ctx.mpf(target_error)
    if all("." not in part and "e" not in part.lower() for pair in formula.b_half for part in pair):
        if verify(formula, dps=dps, tolerance=target_error).passed:
            return formula

    m = formula.m
    words = lyndon_words(formula.order)
    degree = max(map(len, words), default=1)
    rows = [(word_index(word), ctx.one / factorial(len(word))) for word in words]
    free = tuple(formula.b_half_values(ctx)[:-1])
    half = _consistent_half(ctx, free, m)
    if any(value.real <= 0 for value in half):
        raise RefinementError("normalizing the seed gives a nonpositive real B coefficient")

    for iteration in range(max_iterations + 1):
        b = _mirror(half, m)
        series, before_b, after_b = _product_series(ctx, degree, m, b, keep_prefixes=True)
        residual = [series[index] - target for index, target in rows]
        if max(map(abs, residual), default=ctx.zero) <= tolerance:
            break
        if iteration == max_iterations:
            raise RefinementError("reached the iteration limit before meeting the precision target")
        if not free:
            raise RefinementError("the formula has no free coefficients for refinement")
        jacobian = _jacobian(ctx, words, before_b, after_b, m)
        weights = _row_weights(ctx, jacobian)
        scaled_jacobian = ctx.matrix(
            [
                [weight * jacobian[i, j] for j in range(jacobian.cols)]
                for i, weight in enumerate(weights)
            ]
        )
        scaled_residual = [weight * value for weight, value in zip(weights, residual)]
        delta = _newton_step(ctx, scaled_jacobian, scaled_residual)
        residual_norm = ctx.fsum(abs(value) ** 2 for value in scaled_residual)
        scale = ctx.one
        for _ in range(12):
            trial_free = tuple(value + scale * step for value, step in zip(free, delta))
            trial_half = _consistent_half(ctx, trial_free, m)
            if all(value.real > 0 for value in trial_half):
                trial_b = _mirror(trial_half, m)
                trial_series, _, _ = _product_series(ctx, degree, m, trial_b)
                trial_norm = ctx.fsum(
                    abs(weight * (trial_series[i] - target)) ** 2
                    for weight, (i, target) in zip(weights, rows)
                )
                if trial_norm < residual_norm:
                    free, half = trial_free, trial_half
                    break
            scale /= 2
        else:
            raise RefinementError("no feasible Newton step reduced the order-condition residual")

    rounded = tuple(
        (ctx.nstr(z.real, digits, strip_zeros=False), ctx.nstr(z.imag, digits, strip_zeros=False))
        for z in half
    )
    result = Formula(formula.order, m, rounded)
    if not verify(result, dps=dps, tolerance=f"1e{1 - digits}").passed:
        raise RefinementError("the rounded coefficients failed full word verification")
    return result


def _truncate_component(value: str, digits: int) -> str:
    fraction = Fraction(value) if "/" in value else Fraction(Decimal(value))
    if not fraction:
        return "0"
    numerator, denominator = abs(fraction.numerator), fraction.denominator
    exponent = Decimal(numerator).adjusted() - Decimal(denominator).adjusted()
    if exponent >= 0:
        below_power = numerator < denominator * 10**exponent
    else:
        below_power = numerator * 10**-exponent < denominator
    if below_power:
        exponent -= 1
    shift = digits - 1 - exponent
    if shift >= 0:
        truncated = numerator * 10**shift // denominator
    else:
        truncated = numerator // (denominator * 10**-shift)
    return str(Decimal((int(fraction < 0), Decimal(truncated).as_tuple().digits, -shift)))


def _truncate(formula: Formula, *, digits: int) -> Formula:
    """Truncate the stored components towards zero."""
    b_half = tuple(
        (_truncate_component(real, digits), _truncate_component(imag, digits))
        for real, imag in formula.b_half
    )
    return Formula(formula.order, formula.m, b_half)
