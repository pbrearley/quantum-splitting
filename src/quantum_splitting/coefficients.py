"""Read and write coefficient files with exact coefficient strings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from mpmath import nstr

from .model import Formula
from .report import FormulaDiagnostics, formula_diagnostics


def formula_from_dict(data: Mapping[str, Any]) -> Formula:
    """Read the order, number of A factors, and stored coefficient pairs."""
    if not isinstance(data, Mapping):
        raise ValueError("a coefficient file must contain a JSON object")

    try:
        return Formula(order=data["order"], m=data["m"], b_half=data["b_half"])
    except KeyError as exc:
        raise ValueError(f"missing coefficient field {exc.args[0]!r}") from None


def formula_to_dict(formula: Formula, *, seed: int | None = None) -> dict[str, Any]:
    """Return a JSON-ready record with the coefficient strings unmodified."""
    record: dict[str, Any] = {"order": formula.order, "m": formula.m}
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record["seed"] = seed
    record.update(
        free_complex_coefficients=formula.free_complex_coefficients,
        complex_order_conditions=formula.complex_order_conditions,
        b_half=[list(pair) for pair in formula.b_half],
    )
    return record


def load_formula(path: str | Path) -> Formula:
    """Load a formula from the ``order``, ``m``, and ``b_half`` JSON fields."""
    with Path(path).open(encoding="utf-8") as stream:
        return formula_from_dict(json.load(stream))


def save_formula(
    formula: Formula,
    path: str | Path,
    *,
    dps: int = 80,
    include_diagnostics: bool = True,
    seed: int | None = None,
) -> FormulaDiagnostics | None:
    """Atomically write exact coefficients and return the saved diagnostics.

    Diagnostics evaluate the coefficient strings as written at a minimum of ``dps``
    working digits. Metrics are stored as strings in scientific notation with
    up to 16 significant figures. Set ``include_diagnostics=False`` for a fast
    search checkpoint. Pass ``seed`` to record the random search seed.
    """
    record = formula_to_dict(formula, seed=seed)
    result = None
    if include_diagnostics:
        result = formula_diagnostics(formula, dps=dps)
        scientific = partial(
            nstr, n=min(result.dps, 16), min_fixed=0, max_fixed=0, show_zero_exponent=True
        )
        record["diagnostics"] = {
            "dps": result.dps,
            "b_l1": scientific(result.b_l1),
            "b_l2": scientific(result.b_l2),
            "max_word_error": scientific(result.max_word_error),
            "leading_error_l2": scientific(result.leading_error_l2),
            "leading_error_max": scientific(result.leading_error_max),
            "min_abs_real_b": scientific(result.min_abs_real_b),
            "min_abs_imag_b": scientific(result.min_abs_imag_b),
        }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
