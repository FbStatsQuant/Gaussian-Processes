# Gaussian Processes

Implementations and examples of Gaussian Process regression in both Python and R.

## Files

| File | Description |
|------|-------------|
| `GP_Application_1.ipynb` | Python: multi-scale GP regression with a custom Changepoint kernel built from scratch using NumPy/SciPy |
| `GP_Application_1.R` | R: GP regression using a Matérn 5/2 kernel via `DiceKriging` with MLE hyperparameter fitting |
| `lecture.md` | Full lecture notes — GP theory, kernel breakdown, posterior derivation, and code walkthrough |

## Test Function

Both examples fit the same adversarial function:
```
f(x) = sin(x) · exp(-x/5) + 0.5 · sin(5x)
```

It combines a slow decaying sinusoid with a fast oscillation, making it a non-trivial target for any stationary kernel.

## Dependencies

**Python:** `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `GPy`, `gpytorch`

**R:** `DiceKriging`, `kernlab`, `GPfit`, `mlegp`, `ggplot2`, `dplyr`

## Read the Lecture

Start with [`lecture.md`](./lecture.md) if you want to understand what the code is doing and why.
