"""Verify the archived examples and the coefficient file format."""

import json
import re
from pathlib import Path

import pytest
from mpmath import mp

from quantum_splitting import Formula, load_formula, save_formula, verify


@pytest.mark.parametrize(
    ("order", "m", "suffix", "seed"),
    [
        (4, 4, "", 1),
        (6, 16, "", 1114),
        (6, 16, "_minl1", 27),
        (8, 52, "", 1255579),
        (8, 52, "_minl1", 91291490),
        (10, 164, "", 1342945),
        (10, 164, "_minl1", 738222),
    ],
)
def test_archived_example_is_verified(order, m, suffix, seed):
    path = Path(__file__).resolve().parents[1] / "coefficients" / f"order{order}_m{m}{suffix}.json"
    formula = load_formula(path)
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == seed

    assert formula.order == order
    assert formula.m == m
    assert formula.a == (f"1/{m}",) * m
    assert len(formula.b) == m + 1
    assert formula.b == formula.b[::-1]
    report = verify(formula, dps=88, tolerance="1e-63")
    assert report.passed, f"{path.name} failed verification: {report.max_residual}"


@pytest.mark.parametrize("m", [2, 3])
def test_mirroring_respects_a_factor_parity(m):
    outer = ("0.2", "0.1")
    middle = ("0.3", "-0.1")
    formula = Formula(2, m, (outer, middle))

    assert formula.b == (outer,) + (middle,) * (m - 1) + (outer,)


def test_round_trip_preserves_coefficient_strings(tmp_path):
    formula = Formula(2, 2, (("1/3", "-1.45000E-13"), ("0.333000", "+0.00")))
    destination = tmp_path / "coefficients.json"

    save_formula(formula, destination)

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record.pop("diagnostics")["dps"] == 80
    assert record == {
        "order": 2,
        "m": 2,
        "free_complex_coefficients": 1,
        "complex_order_conditions": 0,
        "b_half": [list(pair) for pair in formula.b_half],
    }
    assert load_formula(destination) == formula


def test_binary_float_is_rejected_in_exact_json(tmp_path):
    source = tmp_path / "coefficients.json"
    source.write_text('{"order": 2, "m": 1, "b_half": [[0.5, "0"]]}', encoding="utf-8")

    with pytest.raises(ValueError, match="strings"):
        load_formula(source)


@pytest.mark.parametrize(
    ("record", "missing_field"),
    [
        ({"order": 2, "stages": 1, "b_half": [["1/2", "0"]]}, "m"),
        ({"summary": {"order": 2, "m": 1, "b_independent": [["1/2", "0"]]}}, "order"),
    ],
)
def test_load_requires_canonical_fields(tmp_path, record, missing_field):
    source = tmp_path / "coefficients.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=f"missing coefficient field '{missing_field}'"):
        load_formula(source)


def test_saved_diagnostics_match_strang_formula(tmp_path):
    formula = Formula(2, 1, (("1/2", "0"),))
    destination = tmp_path / "strang.json"

    result = save_formula(formula, destination)

    stored = json.loads(destination.read_text(encoding="utf-8"))["diagnostics"]
    ctx = mp.clone()
    ctx.dps = stored["dps"]
    assert result is not None
    assert result.b_l1 == 1
    assert stored["dps"] == 80
    assert stored["b_l1"] == "1.0e+0"
    assert stored["min_abs_imag_b"] == "0.0e+0"
    assert all(
        re.fullmatch(r"-?\d\.\d{1,15}e[+-]\d+", value)
        for name, value in stored.items()
        if name != "dps"
    )
    assert ctx.mpf(stored["b_l1"]) == ctx.one
    assert ctx.almosteq(ctx.mpf(stored["b_l2"]), ctx.sqrt(ctx.one / 2), rel_eps=1e-15)
    assert ctx.mpf(stored["max_word_error"]) == ctx.zero
    assert ctx.almosteq(ctx.mpf(stored["leading_error_max"]), ctx.one / 6, rel_eps=1e-15)
    assert ctx.almosteq(
        ctx.mpf(stored["leading_error_l2"]), ctx.sqrt(ctx.mpf(5) / 96), rel_eps=1e-15
    )


def test_saved_diagnostics_preserve_small_exponents(tmp_path):
    formula = Formula(2, 1, (("1/2", "1e-400"),))
    destination = tmp_path / "coefficients.json"

    save_formula(formula, destination)

    stored = json.loads(destination.read_text(encoding="utf-8"))["diagnostics"]
    assert stored["min_abs_imag_b"] == "1.0e-400"


def test_checkpoint_can_omit_diagnostics(tmp_path):
    formula = Formula(2, 1, (("1/2", "0"),))
    destination = tmp_path / "checkpoint.json"

    assert save_formula(formula, destination, include_diagnostics=False) is None
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert "diagnostics" not in record
    assert record["free_complex_coefficients"] == 0
    assert record["complex_order_conditions"] == 0
    assert load_formula(destination) == formula
