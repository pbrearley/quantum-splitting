# Splitting Coefficients for Dissipative Dynamics

[![CI](https://github.com/pbrearley/quantum-splitting/actions/workflows/tests.yml/badge.svg)](https://github.com/pbrearley/quantum-splitting/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-guide-blue)](docs/getting-started.md)
[![arXiv](https://img.shields.io/badge/arXiv-2511.19659-b31b1b)](https://arxiv.org/abs/2511.19659)
[![Licence: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Search for coefficients of symmetric operator splittings, optimise them for a chosen objective, and verify them in arbitrary precision. [Paper, arXiv:2511.19659](https://arxiv.org/abs/2511.19659)

The formulas approximate $e^{h(A+B)}$ by alternating factors with $m$ equally spaced $A$ factors and $m+1$ palindromic $B$ factors. In the paper, $A=iH_2$ and $B=H_1$. The $A$ coefficients are $1/m$ and every $B$ coefficient has positive real part.

## Installation

Install from the project root with Python 3.11 or newer and a C++17 compiler.

```bash
python -m pip install .
```

## Example coefficients

The `coefficients/` directory holds the order-4 baseline and pairs for orders 6, 8, and 10. The default filenames identify the low leading-error selections. The `_minl1.json` files prioritise a low coefficient 1-norm. Each archive records its search seed. [Example coefficient index](coefficients/README.md)

## Search for coefficients

The search examples find coefficients from a random start and refine them to 64 significant figures. Run the order-8 search from the project root.

```bash
python examples/03_order8.py
```

The script writes `examples/order8_m52.json`. [Search examples for all four orders](docs/getting-started.md#search-and-optimise-examples)

The manifold optimisation example starts from an order-4 formula with $m=12$ and minimises the coefficient 1-norm, coefficient 2-norm, and leading word error in separate runs. [Optimisation example](examples/05_optimise_manifold.py)

## Verify and refine

Run `python examples/06_verify_coefficients.py` to check the seven archived formulas. Run `python examples/07_refine_coefficients.py` to refine the order-6 formula to 100 significant figures and verify the result.

```bash
quantum-splitting verify coefficients/order8_m52.json --tolerance 1e-63 --leading
quantum-splitting refine coefficients/order8_m52.json --digits 20 \
  -o examples/order8_m52_20digits.json
```

The same commands are available via `python -m quantum_splitting`. [More examples](examples/README.md)

## Plot the example coefficients

Run `python examples/08_plots.py` to compare the coefficient profiles and convergence of the example formulas. [Convergence example](docs/getting-started.md#convergence-in-a-2-by-2-system)

## Documentation

- [Getting started](docs/getting-started.md) covers the CLI, JSON schema, and precision options.
- [Numerical method](docs/method.md) explains the centered order conditions, search, verification, and optimisation.
- [Python API](docs/api.md) describes the public functions and result fields.
- [Examples](examples/README.md) covers search, optimisation, verification, refinement, plotting, and command-line usage.

## Citation

If you use this software or its coefficients, please cite P. Brearley and P. Pfeffer, *High-order splitting of non-unitary operators on quantum computers*, arXiv:2511.19659 (2026). [Citation metadata](CITATION.cff)

## Licence

This project is released under the [MIT License](LICENSE).
