"""Console progress and diagnostics for searches."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from mpmath import mp, nstr

from .model import Formula
from .precision import _word_diagnostics


@dataclass(frozen=True)
class FormulaDiagnostics:
    """Full-sequence norms and word errors measured at ``dps`` working digits."""

    dps: int
    b_l1: mp.mpf
    b_l2: mp.mpf
    max_word_error: mp.mpf
    leading_error_l2: mp.mpf
    leading_error_max: mp.mpf
    min_abs_real_b: mp.mpf
    min_abs_imag_b: mp.mpf


def formula_diagnostics(formula: Formula, *, dps: int = 80) -> FormulaDiagnostics:
    """Evaluate the stored coefficients and all words through degree ``order + 1``.

    The leading errors are the Euclidean norm and largest magnitude of the
    word errors in degree ``order + 1``. The maximum word error covers degrees
    through the target order.
    """
    ctx, errors, leading_l2 = _word_diagnostics(formula, dps)
    assert leading_l2 is not None
    b = formula.b_values(ctx)
    return FormulaDiagnostics(
        dps=ctx.dps,
        b_l1=ctx.fsum(abs(value) for value in b),
        b_l2=ctx.sqrt(ctx.fsum(abs(value) ** 2 for value in b)),
        max_word_error=max(errors[: formula.order]),
        leading_error_l2=leading_l2,
        leading_error_max=errors[-1],
        min_abs_real_b=min(abs(value.real) for value in b),
        min_abs_imag_b=min(abs(value.imag) for value in b),
    )


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


class ProgressReporter:
    """Write a short status line to stderr at most once per ``interval`` seconds.

    ``attempts=0`` means the search has no start limit, so no total or
    time estimate is shown.
    """

    def __init__(
        self,
        order: int,
        m: int,
        attempts: int,
        interval: float,
        *,
        free_coefficients: int,
        order_conditions: int,
    ) -> None:
        self.attempts = attempts
        self.interval = interval
        self.start = self.last = time.monotonic()
        limit = f"maximum starts {attempts}" if attempts else "no limit on starts"
        print(
            f"Searching for an order-{order} formula with m={m} ({limit})\n"
            f"  {free_coefficients} free complex coefficients, "
            f"{order_conditions} complex order conditions",
            file=sys.stderr,
            flush=True,
        )

    def __call__(self, done: int, best_residual: float, verified: bool = True) -> None:
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        self.last = now
        elapsed = now - self.start
        if self.attempts:
            starts = f"{done}/{self.attempts} starts done"
            eta = f", up to ~{_duration(elapsed / done * (self.attempts - done))} left"
        else:
            starts = f"{done} starts done"
            eta = ""
        error = "best error" if verified else "best estimated error"
        print(
            f"  {starts}, {error} {best_residual:.2e}, {_duration(elapsed)} elapsed{eta}",
            file=sys.stderr,
            flush=True,
        )


def print_diagnostics(formula: Formula, dps: int = 80) -> None:
    """Print coefficient norms, word errors and smallest component magnitudes."""
    result = formula_diagnostics(formula, dps=dps)
    print("Diagnostics")
    print(f"  1-norm of b = {nstr(result.b_l1, 10)}")
    print(f"  2-norm of b = {nstr(result.b_l2, 10)}")
    print(f"  Maximum word error through order {formula.order} = {nstr(result.max_word_error, 10)}")
    print(f"  Leading word L2 error = {nstr(result.leading_error_l2, 10)}")
    print(f"  Leading word maximum error = {nstr(result.leading_error_max, 10)}")
    print(f"  min |Re(b)| = {nstr(result.min_abs_real_b, 10)}")
    print(f"  min |Im(b)| = {nstr(result.min_abs_imag_b, 10)}")
