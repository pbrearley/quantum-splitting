"""Compare the archived coefficient profiles and convergence curves."""

from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpmath import mp, nstr

from quantum_splitting import Formula, formula_diagnostics, load_formula

EXAMPLES = Path(__file__).resolve().parent
COEFFICIENTS = EXAMPLES.parent / "coefficients"
PAIRED_FORMULAS = ((6, 16), (8, 52), (10, 164))
SELECTIONS = {"": "Low leading error", "_minl1": "Low coefficient 1-norm"}
FINAL_TIME = 1
STEP_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)
WORKING_DIGITS = 80


def plot_profile(ax: Axes, formula: Formula, heading: str = "") -> None:
    """Plot scaled coefficients alongside the norm and leading word error."""
    # b_values mirrors the stored half. Convert to double precision for plotting.
    with mp.workdps(32):
        b = np.array(formula.b_values(mp), dtype=np.complex128)
    index = np.arange(b.size) / formula.m
    scaled = formula.m * b
    ax.axhline(0, color="0.65", linewidth=0.7)
    ax.plot(index, scaled.real, "o-", color="#17619b", ms=2.5, lw=1, label=r"$\mathrm{Re}(m b_j)$")
    ax.plot(index, scaled.imag, "o-", color="#ad4926", ms=2.5, lw=1, label=r"$\mathrm{Im}(m b_j)$")
    diagnostics = formula_diagnostics(formula, dps=WORKING_DIGITS)
    norm = nstr(diagnostics.b_l1, 4, strip_zeros=False)
    leading_error = nstr(
        diagnostics.leading_error_l2, 4, strip_zeros=False, min_fixed=0, max_fixed=0
    )
    mantissa, _, exponent = leading_error.partition("e")
    if exponent:
        leading_error = rf"{mantissa}\times 10^{{{int(exponent)}}}"
    title = (
        f"Order {formula.order} ($m={formula.m}$)\n"
        rf"$\|b\|_1={norm}$    Leading $L_2$ error $={leading_error}$"
    )
    ax.set_title(f"{heading}\n{title}" if heading else title)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.label_outer()


def plot_profiles(order4: Formula, groups: dict[str, list[Formula]], path: Path) -> None:
    """Plot the order-4 formula centered above the higher-order pairs."""
    fig = plt.figure(figsize=(10, 8.5), dpi=180, layout="constrained")
    grid = fig.add_gridspec(len(PAIRED_FORMULAS) + 1, 4)
    order4_axis = fig.add_subplot(grid[0, 1:3])
    plot_profile(order4_axis, order4)
    order4_axis.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    (left_title, low_error), (right_title, low_norm) = groups.items()
    pairs = zip(low_error, low_norm, strict=True)
    for row, (error_formula, norm_formula) in enumerate(pairs, start=1):
        left = fig.add_subplot(grid[row, :2], sharex=order4_axis)
        right = fig.add_subplot(grid[row, 2:], sharex=order4_axis, sharey=left)
        plot_profile(left, error_formula, left_title if row == 1 else "")
        plot_profile(right, norm_formula, right_title if row == 1 else "")

    legend = order4_axis.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, frameon=False
    )
    legend.set_in_layout(False)
    fig.supxlabel(r"Relative coefficient index $j/m$")
    fig.supylabel(r"Scaled coefficient $m b_j$")

    # Lay out equal-size axes first, then center the order-4 axes and legend together.
    fig.draw_without_rendering()
    left, right = fig.axes[1:3]
    plot_center = (left.get_position().x0 + right.get_position().x1) / 2
    order4_box = order4_axis.get_position()
    legend_box = legend.get_window_extent().transformed(fig.transFigure.inverted())
    group_center = (order4_box.x0 + legend_box.x1) / 2
    fig.set_layout_engine("none")
    order4_axis.set_position(order4_box.translated(plot_center - group_center, 0))

    fig.savefig(path, dpi="figure")
    plt.close(fig)
    print(f"Saved {path}")


def splitting_matrix(b, m, h):
    """Evaluate B0 A B1 ... A Bm for the A and B matrices below."""
    cosine, sine = mp.cos_sin(h / m)
    exp_a = mp.matrix([[cosine, -sine], [sine, cosine]])
    step = mp.diag([1, mp.exp(-h * b[0])])
    for coefficient in b[1:]:
        exp_b = mp.diag([1, mp.exp(-h * coefficient)])
        step = step * exp_a * exp_b
    return step


def convergence_errors(formula: Formula, exact, selection: str) -> list[float]:
    """Compute global errors and report the rate between the two finest steps."""
    coefficients = formula.b_values(mp)
    errors = []
    for steps in STEP_COUNTS:
        h = mp.mpf(FINAL_TIME) / steps
        approximation = splitting_matrix(coefficients, formula.m, h) ** steps
        difference = approximation - exact
        errors.append(mp.sqrt(mp.fsum(abs(entry) ** 2 for entry in difference)))

    rate = mp.log(errors[-2] / errors[-1]) / mp.log(mp.mpf(STEP_COUNTS[-1]) / STEP_COUNTS[-2])
    print(f"{selection} order {formula.order}, m={formula.m}  observed order {float(rate):.3f}")
    return [float(error) for error in errors]


def plot_convergence(common: Formula, groups: dict[str, list[Formula]], path: Path) -> None:
    """Compare the global Frobenius errors on a dissipative 2 by 2 system."""
    fig, axes = plt.subplots(
        1, len(groups), figsize=(10, 4.6), sharex=True, sharey=True, layout="constrained"
    )

    with mp.workdps(WORKING_DIGITS):
        a = mp.matrix([[0, -1], [1, 0]])
        b = mp.diag([0, -1])
        exact = mp.expm(FINAL_TIME * (a + b))
        common_errors = convergence_errors(common, exact, "common")
        for ax, (selection, formulas) in zip(axes, groups.items(), strict=True):
            curves = [(common, common_errors)]
            curves.extend(
                (formula, convergence_errors(formula, exact, selection)) for formula in formulas
            )
            for (formula, errors), marker in zip(curves, cycle("os^D")):
                label = f"Order {formula.order} ($m={formula.m}$)"
                ax.plot(STEP_COUNTS, errors, marker=marker, markersize=4, label=label)

            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(STEP_COUNTS, [str(steps) for steps in STEP_COUNTS])
            ax.set_title(selection)
            ax.grid(color="0.89", linewidth=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    fig.supxlabel(r"Number of splitting steps $N$")
    fig.supylabel(rf"Global error at $T={FINAL_TIME}$ (Frobenius norm)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    print(f"Using coefficients from {COEFFICIENTS}")
    try:
        common = load_formula(COEFFICIENTS / "order4_m4.json")
        groups = {
            title: [
                load_formula(COEFFICIENTS / f"order{order}_m{m}{suffix}.json")
                for order, m in PAIRED_FORMULAS
            ]
            for suffix, title in SELECTIONS.items()
        }
    except FileNotFoundError as error:
        raise SystemExit(f"Missing archived coefficient file {error.filename}") from error
    plot_profiles(common, groups, EXAMPLES / "coefficient_profiles.png")
    plot_convergence(common, groups, EXAMPLES / "convergence.png")


if __name__ == "__main__":
    main()
