# Getting started

## Coefficient files

The `coefficients/` directory holds the order-4 baseline and pairs for orders 6, 8, and 10. The default filenames identify the selections for a low Euclidean norm of the leading word error. The `_minl1.json` suffix identifies the selections for a low 1-norm of the $B$ coefficients. The search scripts write the order-4 baseline and the low-error results in `examples/`. [Example coefficient index](../coefficients/README.md)

```json
{
  "order": 4,
  "m": 4,
  "free_complex_coefficients": 2,
  "complex_order_conditions": 2,
  "b_half": [
    ["1/10", "-1/30"],
    ["4/15", "2/15"],
    ["4/15", "-1/5"]
  ]
}
```

| Symbol | Meaning | JSON or Python |
| --- | --- | --- |
| $p$ | Formal order to check | `order` |
| $m$ | Number of $A$ factors | `m` |
| | Number of free complex $B$ coefficients after symmetry and consistency | `free_complex_coefficients`, `formula.free_complex_coefficients` |
| | Number of complex Lyndon conditions above degree one | `complex_order_conditions`, `formula.complex_order_conditions` |
| $a_j$ | Each $A$ coefficient $1/m$ | Implicit in JSON, `formula.a` in Python |
| $b_0,\ldots,b_{\lfloor m/2\rfloor}$ | Stored real and imaginary pairs in product order | `b_half` |
| $b_0,\ldots,b_m$ | Full sequence, with $b_j=b_{m-j}$ | `formula.b` |
| | Optional integer used to seed the random search | `seed` |

Form the product from left to right, alternating $B$ and $A$ and ending with $B$. Mirror the stored half without conjugation. If $m$ is even, the central $B$ coefficient appears once. If $m$ is odd, all stored values appear twice.

Saved coefficient files include a `diagnostics` object. A word error is the difference between the word's coefficient in the split product and in $e^{h(A+B)}$. Verification recalculates these errors from the stored coefficients.

| Field | Value |
| --- | --- |
| `dps` | Working precision in decimal digits |
| `b_l1`, `b_l2` | L1 and L2 norms of the full $B$ coefficient sequence |
| `max_word_error` | Largest absolute word error across degrees 1 through $p$ |
| `leading_error_l2` | Euclidean norm of all word errors at degree $p+1$ |
| `leading_error_max` | Largest absolute word error at degree $p+1$ |
| `min_abs_real_b`, `min_abs_imag_b` | Smallest $\lvert\operatorname{Re} b_j\rvert$ and $\lvert\operatorname{Im} b_j\rvert$ |

## Commands

| Command | Input | Main options |
| --- | --- | --- |
| `verify` | JSON path | `--dps`, `--tolerance`, `--leading` |
| `refine` | JSON path | `--digits`, `--max-iterations`, `--output` |
| `search` | `--order` and `--m` | `--attempts`, `--max-nfev`, `--seed`, `--initial`, `--min-real`, `--tolerance`, `--workers`, `--output` |
| `optimize` | JSON path | `--objective`, `--min-real`, `--digits`, `--max-iterations`, `--output` |

Use `quantum-splitting COMMAND --help` for options and defaults. Verification prints the maximum absolute word error for each degree. A pass requires that the maximum through the specified order be within tolerance and every $\operatorname{Re} b_j$ be positive. `--leading` also reports the maximum and Euclidean word errors one degree higher.

The verifier checks degree-one consistency and positivity exactly from the stored strings. It uses interval bounds for higher degrees and retries at higher precision if a bound overlaps the tolerance. If the result remains inconclusive, the check fails. Use `--dps` to set the minimum working precision. The [verification example](../examples/06_verify_coefficients.py) checks all seven example files.

`--digits` counts significant figures in each real or imaginary component. After reducing precision, use `verify` to check that the coefficients still meet your tolerance. [Refinement API](api.md#verification-and-refinement)

```bash
quantum-splitting verify coefficients/order8_m52.json --tolerance 1e-63 --leading
quantum-splitting refine coefficients/order8_m52.json --digits 100 -o examples/order8_m52_100digits.json
quantum-splitting refine coefficients/order8_m52.json --digits 20 -o examples/order8_m52_20digits.json
```

The [refinement example](../examples/07_refine_coefficients.py) refines the archived low-error order-6 coefficients at $m=16$ to 100 significant figures. It checks the full-word residual at tolerance `1e-99` and writes `examples/order6_m16_100digits.json`.

## Searching

Search for even order with a specified number $m$ of $A$ factors. `--attempts` limits the number of random starts. Each start has a default budget of 1,600 residual evaluations, adjustable with `--max-nfev`. `--workers` runs starts in separate processes. A fixed `--seed` defines the start sequence across worker counts. Choose `--attempts 0` to continue until a candidate meets the tolerance or you interrupt the process.

The search draws Gaussian real offsets and logistic imaginary offsets, then solves the centered order conditions in double precision. It evaluates the original word errors to decide whether a candidate meets the requested tolerance. A seed identifies a start sequence for this implementation. [Numerical method](method.md#centered-order-conditions)

Before solving, the search reports the number of free complex coefficients and complex order conditions. For this family, the $A$ coefficients are fixed, and symmetry and consistency leave $\lfloor m/2\rfloor$ free complex $B$ coefficients. A search with fewer free coefficients than conditions is rejected and reports the minimum $m$ required by this check. Order 6 with $m=16$ passes because it has eight free complex coefficients and eight complex conditions.

`--initial` uses a saved formula with the same order and $m$ as a starting point. `--min-real` imposes a strict lower bound for every $\operatorname{Re} b_j$. The bound must be nonnegative and less than $1/(m+1)$. The command saves better candidates as the search progresses and adds diagnostics on completion. A search started without `--initial` records its random seed.

For tolerances tighter than `1e-12`, the search attempts high-precision refinement when a candidate's full-word residual is at most `1e-12`. Use `refine` when you already have coefficients that you want to compute to higher precision.

Search for an order-4 formula at high precision.

```bash
quantum-splitting search --order 4 --m 4 --seed 1 --tolerance 1e-63 -o examples/order4_m4.json
```

## Search and optimise examples

Run the scripts from the project root. The first four examples search for formulas with the minimum $m$ admitted by the count of order conditions.

| Order | Script | $m$ | Complex conditions | Output file |
| --- | --- | ---: | ---: | --- |
| 4 | [`01_order4.py`](../examples/01_order4.py) | 4 | 2 | `examples/order4_m4.json` |
| 6 | [`02_order6.py`](../examples/02_order6.py) | 16 | 8 | `examples/order6_m16.json` |
| 8 | [`03_order8.py`](../examples/03_order8.py) | 52 | 26 | `examples/order8_m52.json` |
| 10 | [`04_order10.py`](../examples/04_order10.py) | 164 | 82 | `examples/order10_m164.json` |

Each script uses a fixed seed and saves a formula verified at tolerance `1e-63`. The seed fixes the starting points. Small numerical differences can affect which root the search reaches. Use [the files in `coefficients/`](../coefficients/README.md) when you need the exact archived values. [Low-norm search example](../examples/README.md#console-commands)

The [manifold optimisation example](../examples/05_optimise_manifold.py) uses order 4 with $m=12$. Run `python examples/05_optimise_manifold.py` to search for a starting formula and minimise all three objectives from it. With six free complex coefficients and two complex order conditions, a regular solution has four complex directions available for optimisation.

## Optimising an existing formula

Start with a formula whose full-word residual is at most `1e-12` through the requested order. Each real part must strictly exceed `--min-real`, which defaults to zero.

| `--objective` | Quantity minimised |
| --- | --- |
| `l1` (default) | $\sum_{j=0}^{m} \lvert b_j\rvert$ |
| `l2` | $\left(\sum_{j=0}^{m} \lvert b_j\rvert^2\right)^{1/2}$ |
| `error` | Euclidean norm of the degree-$p+1$ word-coefficient error |

To minimise the leading error from the command line, first search for a starting formula and then pass it to `optimize`.

```bash
quantum-splitting search --order 4 --m 12 --seed 20 -o examples/order4_m12_search.json
quantum-splitting optimize examples/order4_m12_search.json --objective error --digits 64 \
  -o examples/order4_m12_minerror.json
```

`--digits` sets the significant figures for refinement and must be at least 15. The output is checked after rounding against every word through degree $p$ with absolute tolerance $10^{1-\text{digits}}$. The optimised metric is reported on those coefficients. Use `verify --leading` to report both the maximum absolute and Euclidean errors across the degree-$p+1$ words. The `error` objective uses the Euclidean error.

The optimiser periodically recomputes the constraint scaling at the best verified coefficients and continues until local convergence or a pass with no verified improvement. Use `--max-iterations` to set an explicit budget. Different starting formulas can lead to different local solutions and objective values.

## Convergence in a 2 by 2 system

The convergence example uses the noncommuting matrices

$$
A=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\qquad
B=\begin{pmatrix}0&0\\0&-1\end{pmatrix}.
$$

$A$ is skew-Hermitian and $B$ is negative semidefinite. For the $m=4$ baseline and both selections at orders 6, 8, and 10, the script plots $\|S(T/N)^N-e^{T(A+B)}\|_F$ at the fixed final time $T=1$. The horizontal axis counts complete splitting steps. Each step contains $m$ $A$ factors and $m+1$ $B$ factors.

The convergence panels share their scales and use the same order-4 baseline. The script computes the matrix errors at 80-digit working precision and estimates the observed order from the errors at 64 and 128 steps.

The coefficient profiles report the coefficient 1-norm and `leading_error_l2`, the metric used by the optimiser's `error` objective. [Leading word error definition](method.md#optimisation-on-the-solution-manifold)

Run `python examples/08_plots.py` to generate both plots. [Plotting script](../examples/08_plots.py)

## Command-line tool exit status

| Exit status | Meaning |
| --- | --- |
| 0 | Command completed and any requested tolerance or local convergence criterion was met |
| 1 | Verification failed, search missed the tolerance, or the local optimiser stopped early |
| 2 | Invalid arguments, file error, or unsuccessful refinement |
| 130 | Interrupted |

If the optimiser stops early, the output still contains the best verified coefficients found, which may be the seed refined to the requested precision.
