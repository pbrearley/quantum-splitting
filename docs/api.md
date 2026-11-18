# Python API

Import the public names from `quantum_splitting`.

## Coefficients

- `Formula(order, m, b_half)` is an immutable coefficient record with `m` equal $A$ factors and `m + 1` $B$ factors. `a` contains exact rational strings and `b` contains the full mirrored real/imaginary sequence. Use `b_values(ctx)` and `b_half_values(ctx)` with an `mpmath` context to obtain complex values at your chosen precision.
- `formula.free_complex_coefficients` counts the free complex $B$ coefficients after symmetry and consistency. `formula.complex_order_conditions` counts the complex Lyndon conditions above degree one. These read-only properties correspond to the JSON fields of the same names.
- `load_formula(path)` loads a coefficient file. `save_formula(formula, path, dps=80)` writes the coefficients and their computed diagnostics atomically, then returns the diagnostics. Pass `seed` to record the integer random seed in the JSON. Pass `include_diagnostics=False` to save a lightweight checkpoint. See the [coefficient schema](getting-started.md#coefficient-files).
- `formula_diagnostics(formula, dps=80)` evaluates the coefficient norms and word errors without writing a file.

## Search

```python
result = search(
    order=4,
    m=4,
    attempts=20,
    seed=0,
    initial=None,
    min_real=0.0,
    max_nfev=1600,
    tolerance=1e-12,
    workers=1,
    on_improvement=None,
    on_progress=None,
    verbose=True,
    progress_interval=15.0,
)
```

`search` solves the centered order conditions in double precision and checks candidates against the original full-word errors. `SearchResult.formula` holds the best candidate. `residual` is the maximum full-word error through the target order, `attempts` counts starts considered, and `converged` records whether the requested tolerance was reached. If supplied, `on_improvement` receives the current `SearchResult` when the search improves the best candidate. If supplied, `on_progress(done, best)` is called after each unsuccessful start with the number of starts completed and the best `SearchResult` so far. Unless `verbose=False`, `search` also writes a header and a status line (starts done, best error, elapsed time) to stderr at most every `progress_interval` seconds. `print_diagnostics(formula)` prints the coefficient norms, word errors, and the smallest `|Re b|` and `|Im b|`. See [search options](getting-started.md#searching).

## Verification and refinement

```python
report = verify(formula, dps=80, tolerance="1e-12", include_leading=False)
more_digits = refine(formula, digits=100, max_iterations=12)
fewer_digits = refine(formula, digits=20)
```

`VerificationResult` exposes `order`, `dps`, `tolerance`, `max_residual`, `min_real_b`, and `passed`. The reported `dps` is the working precision used, which can be higher than requested. The tuple `residuals_by_degree` starts at degree one. If `include_leading=True`, its final entry is the maximum absolute degree-`order + 1` word error and `leading_l2` is the Euclidean norm of those word errors. Otherwise `leading_l2` is `None`.

`refine` returns a `Formula`. A request below the maximum stored decimal precision discards excess significant figures towards zero. Increased precision uses local refinement, which raises `RefinementError` if it cannot find a feasible solution meeting the requested accuracy. Exact rational input is retained when it already meets the target.

## Solution manifold optimisation

```python
from quantum_splitting import optimize, search, verify

start = search(order=4, m=12, seed=20)
if not start.converged:
    raise RuntimeError("No verified starting formula was found.")
result = optimize(
    start.formula,
    objective="error",
    min_real=0.0,
    digits=64,
)
report = verify(result.formula, dps=90, tolerance="1e-63")
print(result.initial_value, result.value, report.passed)
```

The input must have even order at least two and meet the full-word tolerance of `1e-12` and the strict real-part floor. `objective` is one of `l1`, `l2`, or `error`. `digits` must be an integer of at least 15. By default, the optimiser stops at local convergence or when it finds no further verified improvement. Set `max_iterations` to limit the number of optimiser iterations. [Objective definitions](getting-started.md#optimising-an-existing-formula)

| `OptimizationResult` field | Meaning |
| --- | --- |
| `formula` | Verified coefficients at the requested precision |
| `objective` | Objective name |
| `initial_value` | Objective at the seed |
| `value` | Objective at the returned coefficients |
| `residual` | Maximum absolute word error through the target order, as an arbitrary-precision value |
| `iterations` | Local optimiser iterations |
| `converged` | Whether the local optimiser converged |
| `improved` | Whether the returned coefficients improved the objective |
| `message` | Explanation of the termination |

An optimisation can yield a better verified formula before reaching local convergence. Inspect both `converged` and `improved`.

Invalid parameters or seeds raise `ValueError`. The optimiser raises `OptimizationError` if it cannot prepare a feasible formula at the requested precision.
