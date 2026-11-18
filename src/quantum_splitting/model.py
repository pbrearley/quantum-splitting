"""Exact coefficient storage for symmetric, equidistant splittings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ._words import count_order_conditions

if TYPE_CHECKING:
    from mpmath import mpc, mpf
    from mpmath.ctx_mp import MPContext


Coefficient = tuple[str, str]
_T = TypeVar("_T")

_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_RATIONAL = re.compile(r"[+-]?[0-9]+/[0-9]*[1-9][0-9]*")


def parse_component(text: str, ctx: MPContext) -> mpf:
    """Evaluate a decimal or rational coefficient using ``ctx`` precision."""
    return ctx.mpf(text)


def _mirror(half: tuple[_T, ...], m: int) -> tuple[_T, ...]:
    # When m is even, b_(m/2) occurs once. When m is odd, every entry occurs twice.
    return half + half[: (m + 1) // 2][::-1]


@dataclass(frozen=True)
class Formula:
    r"""Coefficients for ``exp(b_0 B) exp(A/m) ... exp(A/m) exp(b_m B)``.

    Attributes
    ----------
    order : int
        Claimed formal order, which can be checked by ``verify``.
    m : int
        Number of equally spaced A factors. The product has ``m + 1``
        B factors.
    b_half : tuple of tuple of str
        Pairs of real and imaginary strings for ``b_0`` through
        ``b_floor(m/2)``. Each component is a finite decimal or an integer
        fraction with a positive denominator.
    """

    order: int
    m: int
    b_half: tuple[Coefficient, ...]

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order < 1:
            raise ValueError("order must be a positive integer")
        if type(self.m) is not int or self.m < 1:
            raise ValueError("m must be a positive integer")

        size = self.m // 2 + 1
        if not isinstance(self.b_half, (list, tuple)) or len(self.b_half) != size:
            raise ValueError(f"b_half must contain {size} coefficient pairs for m={self.m}")

        pairs = []
        for index, pair in enumerate(self.b_half):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"b_half[{index}] must be a real/imaginary pair")
            for component in pair:
                if not isinstance(component, str) or not (
                    _DECIMAL.fullmatch(component) or _RATIONAL.fullmatch(component)
                ):
                    raise ValueError(
                        f"b_half[{index}] components must be finite decimal or rational strings"
                    )
            pairs.append((pair[0], pair[1]))
        object.__setattr__(self, "b_half", tuple(pairs))

    @property
    def a(self) -> tuple[str, ...]:
        """Return the ``m`` copies of ``1/m`` as exact rational strings."""
        return (f"1/{self.m}",) * self.m

    @property
    def b(self) -> tuple[Coefficient, ...]:
        """Return all ``m + 1`` B coefficients in product order."""
        return _mirror(self.b_half, self.m)

    @property
    def free_complex_coefficients(self) -> int:
        """Count the complex B coefficients free after symmetry and consistency."""
        return self.m // 2

    @property
    def complex_order_conditions(self) -> int:
        """Count the complex Lyndon conditions above degree one through ``order``."""
        return count_order_conditions(self.order)

    def b_half_values(self, ctx: MPContext) -> tuple[mpc, ...]:
        """Convert the stored half to complex numbers at ``ctx`` precision."""
        return tuple(
            ctx.mpc(parse_component(real, ctx), parse_component(imag, ctx))
            for real, imag in self.b_half
        )

    def b_values(self, ctx: MPContext) -> tuple[mpc, ...]:
        """Convert all B coefficients to complex numbers at ``ctx`` precision."""
        return _mirror(self.b_half_values(ctx), self.m)
