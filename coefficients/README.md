# Example coefficients

The order-4 formula at $m=4$ serves as a shared baseline. Orders 6, 8, and 10 each have a default selection for low leading word L2 error and a `_minl1.json` selection for a low coefficient 1-norm.

| File | Order | $m$ | Leading word L2 error | Coefficient 1-norm |
| --- | ---: | ---: | ---: | ---: |
| [order4_m4.json](order4_m4.json) | 4 | 4 | `3.753708642765461e-3` | `1.140436638011169e+0` |
| [order6_m16.json](order6_m16.json) | 6 | 16 | `2.244715789951858e-6` | `1.251752031623849e+0` |
| [order6_m16_minl1.json](order6_m16_minl1.json) | 6 | 16 | `1.324486851277987e-5` | `1.144859078056408e+0` |
| [order8_m52.json](order8_m52.json) | 8 | 52 | `8.957337459971833e-11` | `1.250795057472612e+0` |
| [order8_m52_minl1.json](order8_m52_minl1.json) | 8 | 52 | `3.774030582783472e-8` | `1.117928597697871e+0` |
| [order10_m164.json](order10_m164.json) | 10 | 164 | `3.562281749105686e-14` | `1.28119228066367e+0` |
| [order10_m164_minl1.json](order10_m164_minl1.json) | 10 | 164 | `2.131559481537362e-12` | `1.193940212726777e+0` |

The `seed` field records the random seed used in the search. The scripts use the seeds for the default selections. Use the `--seed` option to try the corresponding low-norm search. [Command-line example](../examples/README.md#console-commands)

The `b_half` field stores real and imaginary parts as strings. The full sequence consists of the stored entries followed by their reflection. For even $m$, the middle coefficient appears once. For odd $m$, every stored entry appears twice. The $A$ coefficients are implicit and all equal $1/m$. [JSON format](../docs/getting-started.md#coefficient-files)

Run `python examples/06_verify_coefficients.py` to check the order conditions and positivity of all seven formulas. Run `python examples/08_plots.py` to compare the coefficient profiles and convergence. [Plotting example](../examples/08_plots.py)
