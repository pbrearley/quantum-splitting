# Numerical method

## Symmetry and consistency

The search uses the palindromic composition

$$
S(h)=e^{h b_0B}e^{h A/m}e^{h b_1 B}\cdots e^{h A/m}e^{h b_m B},
\qquad b_j=b_{m-j}.
$$

The $A$ coefficients sum to one by construction. With $w_j=2$ for a mirrored pair and $w_j=1$ for a central $B$ factor, the independent $B$ coefficients obey $\sum_j w_j b_j=1$. The numerical solvers enforce this linear constraint in their coordinates, leaving the order equations and the real-part feasibility constraint.

## Centered order conditions

Let $t$ be the cumulative $A$ time, with $0\le t\le 1$, and let

$$
q(t)=\sum_{\ell=0}^{j} b_\ell-t,
\qquad \frac{j}{m}<t<\frac{j+1}{m}.
$$

The offset $q$ is the difference between the cumulative $B$ time and the cumulative $A$ time. Symmetry and consistency give $q(1-t)=-q(t)$. The search uses the scaled midpoint values $z_j=m\,q((j+1/2)/m)$. Within interval $j$, with $t=(j+u)/m$, the scaled offset is $m q(t)=z_j+1/2-u$.

Writing $X=A+B$ and $Y=B$, factor out the cumulative $Y$ motion and center the constant $X$ evolution at $t=1/2$. This gives the centered product $e^{-hX/2}S(h)e^{-hX/2}$, with the identity as its target. The remaining evolution has letters proportional to $\operatorname{ad}_X^r\operatorname{ad}_Y^k X$ of weight $k+r+1$, with $k\ge 1$. These letters freely generate the commutator ideal in the Lie algebra on $X$ and $Y$, by eliminating $Y$ and then $X$.

The letter integrands are proportional to $(t-1/2)^r q(t)^k$. The solver uses the shifted Legendre polynomial $P_r(2t-1)$ in place of $(t-1/2)^r$. This is a triangular change involving lower weights, whose conditions already vanish when solving for the next weight.

The centered product is palindromic, so its formal logarithm contains only odd weights. The order conditions set the odd-weight Lyndon coefficients from weight 3 through $p-1$ to zero. Each integrand on an $A$ interval is polynomial in the local time. The C++ evaluator forms the polynomial coefficients directly, integrates them term by term, and concatenates the resulting interval integrals. The Jacobian differentiates the same integrals with respect to the offsets.

The search first eliminates the linear moment conditions. For each random start it draws independent Gaussian real offsets and independent logistic imaginary offsets, projects them onto the remaining free coordinates, and scales the real perturbation to respect the real-part bound. The imaginary offsets are adjusted to satisfy the lowest nonlinear moment when the free coordinates permit it.

The local iteration normalises the Jacobian rows and uses a singular value decomposition to solve the damped linearised system, with extra weight on small singular modes. It corrects steps that approach the real-part bound and uses a line search to reduce the scaled residual. At higher orders it first fits progressively larger subsets of the centered conditions, then solves the full system. The full-word verifier checks the resulting coefficients in arbitrary precision.

The centered transformation preserves the order conditions while improving the conditioning of the local solve. Distinct starts can converge to different roots. The rule for drawing the offsets and the numerical details of the local iteration both affect which root a given seed reaches.

## Full-word verification and refinement

In the original alphabet $A<B$, every word of length $d$ has coefficient $1/d!$ in the degree-$d$ term of $e^{h(A+B)}$. Palindromic reversal gives $S(-h)=S(h)^{-1}$, so the formal logarithm of $S$ contains only odd degrees. After consistency, the first possible discrepancy is an odd-degree homogeneous Lie polynomial. The Lyndon Lie basis expands triangularly into associative words. Once the lower degrees match, vanishing of the Lyndon word errors at the next odd degree forces every word error at that degree to vanish.

The verifier checks consistency and positivity using exact arithmetic on the stored coefficient strings. It expands the full product with guard digits and bounds the higher-degree errors using outward-rounded interval arithmetic. A pass requires each bound through degree $p$ to meet the tolerance. The optional degree-$p+1$ result measures the error at the next degree and is excluded from the pass criterion.

At increased precision, the refiner takes a nearby seed, enforces consistency, and iterates on the original odd-degree Lyndon word conditions. It checks the serialized result against the full-word conditions and checks real-part positivity. If there are free directions, a rounded formula can be close to multiple higher-precision solutions.

The number of formal words grows exponentially with the order. The examples use the same verification procedure for orders 4, 6, 8, and 10.

## Optimisation on the solution manifold

Write $F(y)$ for the original Lyndon word conditions in the free coordinates $y$. At a regular solution, the null space of $DF(y)$ gives the directions in which coefficients can move while satisfying the order equations to first order. In the order-4 example with $m=12$, the Jacobian has full row rank and leaves four complex directions. Optimisation adds a scalar objective to the order equations and imposes the real-part floor on each $B$ coefficient.

Guard digits limit cancellation errors when nearly dependent order equations are combined. The solver uses a singular value decomposition of the initial Jacobian to scale the equations before solving the real constrained problem with sequential least squares programming. [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

For a target order $p$, the `error` objective is the Euclidean norm of the leading word errors.

$$
\left(\sum_{w\in\{A,B\}^{p+1}}\left\lvert [w]S-\frac{1}{(p+1)!}\right\rvert^2\right)^{1/2},
$$

where $[w]S$ is the word coefficient in the degree-$p+1$ term of the product. This is a norm in the formal word basis. Evaluating the same error on a pair of concrete operators can give a different ranking of formulas.

The solver refines candidates in arbitrary precision as needed and checks the returned coefficients against every word through the target order. It also evaluates the objective on the returned coefficients. A fixed order and $m$ can admit multiple local optima for the same objective. Comparing verified results from different starting formulas helps find smaller objective values.
