"""Exercise the public API calls in a copy of the standalone examples."""

import json
import runpy
from pathlib import Path
from shutil import copyfile

import pytest

import quantum_splitting as ds
from quantum_splitting._cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def test_square_search_saves_one_formula_without_optimisation(monkeypatch, tmp_path, capsys):
    seed_formula = ds.load_formula(ROOT / "coefficients/order4_m4.json")
    script = tmp_path / "01_order4.py"
    copyfile(EXAMPLES / script.name, script)

    def search(*, order, m, seed, attempts, workers, min_real, tolerance):
        assert (order, m, seed, attempts, workers, min_real, tolerance) == (
            4,
            4,
            1,
            100,
            1,
            1e-10,
            1e-63,
        )
        return ds.SearchResult(formula=seed_formula, residual=0.0, attempts=1, converged=True)

    def optimize(*args, **kwargs):
        pytest.fail("A square search should save the search result directly")

    monkeypatch.setattr(ds, "search", search)
    monkeypatch.setattr(ds, "optimize", optimize)
    monkeypatch.chdir(tmp_path.parent)
    runpy.run_path(str(script), run_name="__main__")

    path = tmp_path / "order4_m4.json"
    assert list(tmp_path.glob("*.json")) == [path]
    assert ds.load_formula(path) == seed_formula
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 1
    assert f"Saved {path.resolve()}" in capsys.readouterr().out


def test_failed_search_skips_optimisation_and_output(monkeypatch, tmp_path):
    script = tmp_path / "04_order10.py"
    copyfile(EXAMPLES / script.name, script)
    fallback = ds.Formula(10, 1, (("1/2", "0"),))

    def search(**options):
        assert options["m"] == 164
        return ds.SearchResult(formula=fallback, residual=0.1, attempts=10, converged=False)

    def optimize(*args, **kwargs):
        pytest.fail("Optimisation requires a verified search result")

    monkeypatch.setattr(ds, "search", search)
    monkeypatch.setattr(ds, "optimize", optimize)
    with pytest.raises(SystemExit, match="No verified order-10 formula"):
        runpy.run_path(str(script), run_name="__main__")
    assert not list(tmp_path.glob("*.json"))
    assert not (tmp_path / "coefficients").exists()


def test_optimisation_example_saves_each_objective_with_the_same_seed(
    monkeypatch, tmp_path, capsys
):
    script = tmp_path / "05_optimise_manifold.py"
    copyfile(EXAMPLES / script.name, script)
    seed = ds.Formula(4, 12, (("1/13", "0"),) * 7)
    l1_formula = ds.Formula(seed.order, seed.m, (("1/13", "0.01"), *seed.b_half[1:]))
    output_dir = tmp_path
    calls = []

    def search(**options):
        assert (options["order"], options["m"], options["seed"]) == (4, 12, 20)
        return ds.SearchResult(seed, 0.0, 1, True)

    def optimize(formula, *, objective, digits):
        assert digits == 64
        assert formula == seed
        if calls:
            previous = calls[-1]
            path = output_dir / f"order4_m12_min{previous}.json"
            assert ds.load_formula(path) == (l1_formula if previous == "l1" else seed)
            assert f"Saved {path.resolve()}" in capsys.readouterr().out
        calls.append(objective)
        answer = l1_formula if objective == "l1" else formula
        return ds.OptimizationResult(answer, objective, 1.0, 1.0, 0.0, 1, True, False, "Converged")

    monkeypatch.setattr(ds, "search", search)
    monkeypatch.setattr(ds, "optimize", optimize)
    runpy.run_path(str(script), run_name="__main__")

    assert calls == ["l1", "l2", "error"]
    assert {path.name for path in output_dir.glob("*.json")} == {
        "order4_m12_minl1.json",
        "order4_m12_minl2.json",
        "order4_m12_minerror.json",
    }
    assert ds.load_formula(output_dir / "order4_m12_minerror.json") == seed
    assert (
        json.loads((output_dir / "order4_m12_minerror.json").read_text(encoding="utf-8"))["seed"]
        == 20
    )
    assert f"Saved {output_dir.resolve() / 'order4_m12_minerror.json'}" in capsys.readouterr().out


def test_verification_example_exits_unsuccessfully_when_an_early_file_fails(tmp_path, capsys):
    examples = tmp_path / "examples"
    coefficients = tmp_path / "coefficients"
    examples.mkdir()
    coefficients.mkdir()
    script = examples / "06_verify_coefficients.py"
    copyfile(EXAMPLES / script.name, script)
    for source in (ROOT / "coefficients").glob("*.json"):
        copyfile(source, coefficients / source.name)

    path = coefficients / "order4_m4.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["b_half"][0][0] = "0.125"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_status.value.code == 1
    output = capsys.readouterr().out
    assert "FAIL  order4_m4.json" in output
    assert "PASS  order6_m16.json" in output
    assert "PASS  order10_m164_minl1.json" in output
    assert len(output.splitlines()) == 7


@pytest.mark.parametrize("use_initial", [False, True])
def test_search_command_records_seed_for_a_random_start(tmp_path, use_initial):
    destination = tmp_path / "out.json"
    args = ["search", "--order", "2", "--m", "1", "--seed", "0", "-o", str(destination)]
    if use_initial:
        source = tmp_path / "initial.json"
        ds.save_formula(ds.Formula(2, 1, (("1/2", "0"),)), source, seed=7)
        args.extend(["--initial", str(source)])

    assert main(args) == 0
    record = json.loads(destination.read_text(encoding="utf-8"))
    if use_initial:
        assert "seed" not in record
    else:
        assert record["seed"] == 0
