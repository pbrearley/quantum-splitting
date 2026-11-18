# Examples

Run these scripts from the project root after installing the package.

| Script | What it does |
| --- | --- |
| [01_order4.py](01_order4.py) | Search for the order-4 baseline with $m=4$. |
| [02_order6.py](02_order6.py) | Search for the low-error order-6 formula with $m=16$. |
| [03_order8.py](03_order8.py) | Search for the low-error order-8 formula with $m=52$. |
| [04_order10.py](04_order10.py) | Search for the low-error order-10 formula with $m=164$. |
| [05_optimise_manifold.py](05_optimise_manifold.py) | Search at order 4 with $m=12$ and optimise along the free directions. |
| [06_verify_coefficients.py](06_verify_coefficients.py) | Verify the seven files in [coefficients/](../coefficients/README.md). |
| [07_refine_coefficients.py](07_refine_coefficients.py) | Refine the archived low-error order-6 formula to 100 significant figures and verify it. |
| [08_plots.py](08_plots.py) | Compare the archived coefficient profiles and convergence side by side. |

The search scripts write `order4_m4.json`, `order6_m16.json`, `order8_m52.json`, and `order10_m164.json`, respectively. Each file records the seed used by the script. [Search settings](../docs/getting-started.md#search-and-optimise-examples)

## Console commands

Run the order-8 search with seed `9870`, from `coefficients/order8_m52_minl1.json`, to reproduce the low-norm selection.

```bash
quantum-splitting search --order 8 --m 52 --seed 9870 --min-real 1e-10 --tolerance 1e-63 \
  -o examples/order8_m52_minl1.json
```

Check the order conditions of one file and report its leading error.

```bash
quantum-splitting verify coefficients/order10_m164.json --dps 84 --tolerance 1e-63 --leading
```

Use `--digits` to increase or reduce the coefficient precision.

```bash
quantum-splitting refine coefficients/order10_m164.json --digits 100 -o examples/order10_m164_100digits.json
quantum-splitting refine coefficients/order10_m164.json --digits 30 -o examples/order10_m164_30digits.json
```

Use `python -m quantum_splitting` in place of `quantum-splitting` to run the same commands.
