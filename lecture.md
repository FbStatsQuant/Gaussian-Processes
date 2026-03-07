# Gaussian Processes — A Lecture from the Code Up

This document explains everything happening in `GP_Application_1.ipynb` and `GP_Application_1.R`. It is written to be read alongside the code, not instead of it.

---

## 1. What Is a Gaussian Process?

A **Gaussian Process (GP)** is a probability distribution over *functions*. Just as a multivariate Gaussian distribution defines a joint distribution over a finite vector of numbers, a GP defines a joint distribution over an infinite collection of function values — one for every possible input `x`.

Formally, we write:

$$ f(x) \sim \mathcal{GP}(m(x), k(x, x')) $$

- `m(x)` is the **mean function** — the expected value of `f` at input `x`. In both files, this is set to a constant (`~1` in R, implicitly zero in Python), meaning we don't assume any particular trend a priori.
- `k(x, x')` is the **covariance function** (or **kernel**) — the key object. It encodes our belief about how correlated `f(x)` and `f(x')` are, which in practice means: *how smooth do we expect the function to be, and over what length scales?*

The GP prior says: *before seeing any data, I believe the function values at any finite set of points follow a joint Gaussian distribution with mean `m` and covariance matrix `K` whose `(i,j)` entry is `k(x_i, x_j)`.*

---

## 2. The Generative Model

Both files use a **noisy regression** setup:

$$ y = f(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2) $$

where `f` is the latent (unobserved) function and `σ²` is observation noise variance. Specifically, the true function is:

```python
# Python
f(x) = sin(x) * exp(-x / 5) + 0.5 * sin(5x)

# R (same structure, different amplitude scaling)
f(x) = sin(x) * 2 * exp(-x / 5) + 0.5 * sin(5x)
```

This function is deliberately adversarial. It combines:

1. A **slow sinusoid** with a decaying amplitude envelope — low-frequency, non-stationary.
2. A **fast oscillation** at 5× the base frequency — high-frequency, stationary.

A single stationary kernel (like RBF) will struggle here. If you tune the length-scale to capture the fast oscillation, the slow component blurs out. If you tune for the slow component, the fast oscillation gets smoothed over entirely. This is a core motivation for the Changepoint GP in the Python notebook.

---

## 3. The Kernel is Everything

The kernel determines the geometry of functions the GP can represent. Here are the kernels used across both files.

### 3.1 Radial Basis Function (RBF) / Squared Exponential

$$ k_{RBF}(x, x') = \sigma_f^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right) $$

- `ℓ` (length-scale): controls how quickly the function varies. Large `ℓ` → smooth, slow-varying. Small `ℓ` → wiggly, fast-varying.
- `σ_f²` (signal variance): controls the marginal variance, i.e., the amplitude of function values.

RBF produces infinitely differentiable (very smooth) functions. It is the default kernel in most GP tutorials, but it is *stationary* — the correlation between two points depends only on their distance `|x - x'|`, not on where in the input space they sit.

### 3.2 Matérn 5/2 (used in the R file)

$$ k_{Matern52}(x, x') = \sigma_f^2 \left(1 + \frac{\sqrt{5}|r|}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}|r|}{\ell}\right) $$

where `r = x - x'`.

This is the kernel used in `km(..., covtype = "matern5_2")` in the R file. Compared to RBF:

- Functions drawn from a Matérn 5/2 GP are **twice differentiable** (compared to infinitely for RBF). This is usually more realistic for real physical processes.
- The correlation drops off more sharply at large distances — less prone to the "over-smooth" pathology of RBF.
- It is still **stationary**: only distance matters, not location.

In practice, Matérn 5/2 is the workhorse kernel for emulation and surrogate modelling and is usually preferred over RBF unless you have a specific reason to believe your function is very smooth.

### 3.3 The Changepoint (CP) Kernel (used in the Python file)

This is a **non-stationary** kernel constructed by blending two RBF kernels using a sigmoid weight:

$$ \sigma(x) = \frac{1}{1 + \exp(-s(x - c))} $$

where `c` is the changepoint location and `s` controls the sharpness of the transition.

The kernel is:

$$ k_{CP}(x, x') = \sigma(x)\sigma(x')k_{RBF1}(x, x') + (1 - \sigma(x))(1 - \sigma(x'))k_{RBF2}(x, x') $$

- For `x << c`: `σ(x) ≈ 1`, so the kernel behaves like `k_RBF1` with parameters `(ℓ₁, σ_f1)`.
- For `x >> c`: `σ(x) ≈ 0`, so the kernel behaves like `k_RBF2` with parameters `(ℓ₂, σ_f2)`.
- Near `c`: the two kernels blend smoothly.

This lets the GP have **different length-scales and signal variances on either side of the changepoint** — exactly what is needed for the test function, which has a decaying amplitude and competing frequencies.

The Python code implements this from scratch in NumPy rather than using a library. This is instructive: it shows you what a GP really is, mechanically.

---

## 4. The GP Posterior: Bayesian Updating

Once we have observations `(X_obs, y_obs)`, we condition the GP prior on them to get the **GP posterior** — an updated distribution over functions consistent with the data.

For a new test point `x*`, the posterior predictive distribution is Gaussian with:

$$ \mu(x^*) = k(x^*, X)[K(X,X) + \sigma^2I]^{-1}y $$
$$ \sigma^2(x^*) = k(x^*, x^*) - k(x^*, X)[K(X,X) + \sigma^2I]^{-1}k(X, x^*) $$

where:
- `K(X, X)` is the `n × n` kernel matrix over training points.
- `k(x*, X)` is the `1 × n` vector of covariances between the test point and training points.
- `σ²I` is the diagonal noise term added to the kernel matrix.

**The posterior mean** is a weighted sum of the observations, where the weights encode how much each training point "influences" the prediction at `x*` — determined entirely by the kernel.

**The posterior variance** quantifies uncertainty. Where the training data is dense, the posterior variance is low (the ±2σ ribbon is tight). Where there is no data, the variance approaches the prior variance (the ribbon widens).

The Python notebook computes this explicitly:

```python
K_train = cp_kernel(x_obs, x_obs, ...) + sigma_noise**2 * np.eye(n_obs)
alpha   = solve_triangular(L_train.T, solve_triangular(L_train, y_obs, lower=True))
mu_cp   = K_star @ alpha
std_cp  = sqrt(diag(K_ss) - sum(v**2, axis=0))
```

The R file delegates this to `DiceKriging`'s `predict()` method, which does the same thing internally.

---

## 5. Why Cholesky? Numerical Stability

The naive way to compute the GP posterior is to directly invert `[K + σ²I]`. This is a bad idea numerically. Kernel matrices are often ill-conditioned (eigenvalues span many orders of magnitude), so direct inversion amplifies floating-point errors.

The standard approach is **Cholesky decomposition**:

$$ K + \sigma^2I = LL^\top \quad (L \text{ is lower triangular}) $$

Then:
- `[K + σ²I]⁻¹·y` is solved as two triangular solves: `L·v = y` then `Lᵀ·α = v`.
- The log-determinant `log|K + σ²I| = 2·Σᵢ log Lᵢᵢ` is cheap to compute.

The Python code does this explicitly with `scipy.linalg.cholesky` and `solve_triangular`. Both steps are `O(n³)` but with the smallest possible constants.

**Computational complexity** is the main practical limitation of GPs. Fitting scales as `O(n³)` and prediction scales as `O(n²)`. With `n = 200` observations (as in both files), this is trivially fast. At `n = 10,000`, it becomes painful. At `n = 100,000`, you need sparse approximations (inducing points, etc.).

---

## 6. Hyperparameter Optimization via Log-Marginal Likelihood

A GP has hyperparameters — the kernel parameters `(ℓ, σ_f)` and noise variance `σ²`. How do we set them?

The **log-marginal likelihood (LML)** is the log-probability of the observed data under the GP model, marginalised over the function values:

$$ \log p(y | X, \theta) = -\frac{1}{2}y^\top[K + \sigma^2I]^{-1}y - \frac{1}{2}\log|K + \sigma^2I| - \frac{n}{2}\log(2\pi) $$

Three terms:
1. **Data fit** (`-½·yᵀ·[K + σ²I]⁻¹·y`): rewards kernels that assign high probability to the observed `y`.
2. **Complexity penalty** (`-½·log|K + σ²I|`): penalises kernels that are overly flexible. This is automatic Occam's Razor — you do not need to set it by hand.
3. **Normalisation constant**: irrelevant for optimisation.

The Python notebook implements this as `neg_log_marginal_likelihood()` and minimises it using L-BFGS-B over the log-transformed hyperparameters `[ℓ₁, σ_f1, ℓ₂, σ_f2]`. Log-transforming ensures the parameters remain positive throughout optimisation.

The R file uses `nugget.estim = TRUE` in `km()`, which tells DiceKriging to estimate the noise variance (nugget) as part of the same MLE optimisation.

**Note on identifiability**: The changepoint location `c_cp = 5.0` is fixed in the Python code (at the midpoint of the domain). In principle it can also be optimised, but the LML landscape over `c` tends to be multimodal, so it is often easier to fix it based on domain knowledge or do a grid search.

---

## 7. Universal Kriging vs Simple Kriging

In the R file:

```r
gp_fit <- km(formula = ~1, ...)
gp_pred <- predict(gp_fit, ..., type = "UK")
```

The `formula = ~1` specifies a **constant mean function** to be estimated from the data. The `type = "UK"` requests **Universal Kriging** predictions.

- **Simple Kriging (SK)**: assumes the mean function is known (typically zero). The posterior formula above is exact.
- **Ordinary Kriging (OK)**: assumes a constant but *unknown* mean. It is estimated implicitly via BLUP (Best Linear Unbiased Predictor).
- **Universal Kriging (UK)**: generalises to an unknown mean that is a linear combination of known basis functions — e.g., `m(x) = β₀ + β₁x`. With `formula = ~1`, this reduces to an unknown constant, equivalent to Ordinary Kriging.

For most practical regression problems, UK/OK is safer than SK because you don't have to specify the global mean level correctly. The uncertainty in the mean is properly propagated into the prediction intervals.

---

## 8. Stationarity vs Non-Stationarity

**Stationary kernel**: `k(x, x')` depends only on `r = x - x'`. The process has the same statistical properties everywhere.

**Non-stationary kernel**: `k(x, x')` depends on the absolute positions `x` and `x'`, not just their difference. Different parts of the input space can have different length-scales, amplitudes, or correlation structures.

The test function in both files is non-stationary: the fast decaying component `sin(5x)·exp(-x/5)` has large amplitude near `x=0` and near-zero amplitude near `x=10`. A stationary GP with a single length-scale cannot capture this.

The R file uses Matérn 5/2 (stationary) — it will do a reasonable job overall but will over-smooth near `x=0` (where the fast oscillation dominates) or under-smooth near `x=10` (where the function is nearly zero). With 200 observations the fit is still numerically decent, but the UQ (uncertainty bands) will be miscalibrated.

The Python notebook uses the CP-GP (non-stationary) — by allowing `ℓ₁ ≠ ℓ₂` and `σ_f1 ≠ σ_f2` on either side of `c=5`, it can adapt to the changing character of the function.

---

## 9. What the Plots Tell You

### True function plot
Shows `f(x)` (black), the noisy observations (red/blue points), and a `±2σ_noise` ribbon around the truth. This ribbon represents irreducible noise — even a perfect model cannot do better than this.

### GP fit plot
Shows:
- **Blue/red line**: posterior mean — the GP's best guess for `f(x)` at each point.
- **Shaded ribbon**: `mean ± 2·posterior_std`. This is the GP's *epistemic* uncertainty (uncertainty about `f`), not observation noise. With 200 densely-placed observations, this ribbon should be tight everywhere in the domain.
- **MAE/MSE annotations**: mean absolute and mean squared error between posterior mean and true function (not noisy observations). This measures how well the GP recovers the latent `f`.

If the GP is well-calibrated, approximately 95% of true function values should fall inside the `±2σ` ribbon.

---

## 10. Library Overview

| Library | Language | What it does |
|---|---|---|
| `DiceKriging` | R | Full GP regression with MLE hyperparameter fitting; UK/SK/OK kriging |
| `kernlab` | R | GP classification and regression; imported but not used directly here |
| `GPfit` | R | GP regression focused on computer experiments (space-filling designs) |
| `mlegp` | R | MLE-based GP fitting, particularly for deterministic computer experiments |
| `sklearn.gaussian_process` | Python | Standard GP regression/classification; good for moderate `n` |
| `GPy` | Python | Full-featured GP library from Sheffield; extensive kernel library, sparse GPs |
| `gpytorch` | Python | GPU-accelerated GPs on top of PyTorch; scales to large `n` via inducing points |
| `scipy.linalg` | Python | Used here to implement the Cholesky-based posterior directly |

The Python notebook imports `GPy`, `torch`, and `gpytorch` but does not use them — the entire CP-GP is implemented from scratch in NumPy. This is intentional: implementing the posterior manually makes the mechanics transparent.

---

## 11. Summary of What the Code Demonstrates

| Concept | Where |
|---|---|
| Noisy GP regression with Matérn 5/2 | `GP_Application_1.R` |
| MLE hyperparameter fitting (nugget estimation) | `GP_Application_1.R` — `nugget.estim = TRUE` |
| Universal Kriging posterior | `GP_Application_1.R` — `type = "UK"` |
| Multi-scale, non-stationary test function | Both files |
| Custom non-stationary (Changepoint) kernel | `GP_Application_1.ipynb` — `cp_kernel()` |
| Log-marginal likelihood optimisation from scratch | `GP_Application_1.ipynb` — `neg_log_marginal_likelihood()` |
| Cholesky-based posterior computation | `GP_Application_1.ipynb` — `cholesky` / `solve_triangular` |
| MAE/MSE evaluation against ground truth | Both files |

---

## 12. Things to Be Aware Of (Practical Gotchas)

**The changepoint location is fixed, not learned.** Setting `c_cp = 5.0` works because the domain is `[0, 10]` and the function's character visibly changes around that point. In practice you would either cross-validate over candidate values, include `c` in the LML optimisation (with care, since the objective is non-convex in `c`), or use a fully Bayesian treatment.

**`np.clip(..., 0, None)` in the variance computation.** Numerical errors in floating point can push the diagonal of the posterior covariance matrix slightly negative. The clip prevents `sqrt` of a negative number. If this happens a lot, your kernel matrix is poorly conditioned — add more noise or use a larger nugget.

**`200` observations is the sweet spot for this demo.** The Cholesky is `O(n³)`, so tripling `n` to 600 multiplies runtime by ~27. At `n = 2000` you would want to switch to sparse or inducing-point approximations.

**L-BFGS-B on log-transformed parameters.** Optimising `log(ℓ)` instead of `ℓ` directly ensures positivity and improves the conditioning of the objective. This is a standard trick — always do it.

**The R and Python functions are not identical.** The R version has a factor of `2` in front of `exp(-x/5)`. The models are fit to different instantiations of the same noisy process (different `sigma_noise = 0.4` vs `0.3`). Do not expect identical results.