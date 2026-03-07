# ── Install packages ──────────────────────────────────────────────────────────
install.packages(c(
  "kernlab",      # GP regression (gausspr)
  "DiceKriging",  # GP modelling and kriging
  "GPfit",        # GP fitting and prediction
  "mlegp",        # Maximum likelihood estimation of GP
  "ggplot2",      # Plotting
  "dplyr"         # Data manipulation
))

# ── Load libraries ────────────────────────────────────────────────────────────
library(kernlab)
library(DiceKriging)
library(GPfit)
library(mlegp)
library(ggplot2)
library(dplyr)

# ── Data generation ───────────────────────────────────────────────────────────
# f(x) = sin(x) * exp(-x/5) + 0.5 * sin(5x)
# y = f(x) + eps,  eps ~ N(0, sigma^2)

f <- function(x) sin(x) * exp(-x / 5) + 0.5 * sin(5 * x)

sigma_noise <- 0.2
set.seed(42)

# Dense grid for the true function
x_plot <- seq(0, 10, length.out = 500)
y_plot <- f(x_plot)

# Noisy observations
n_obs <- 200
x_obs <- sort(runif(n_obs, 0, 10))
y_obs <- f(x_obs) + rnorm(n_obs, 0, sigma_noise)

# ── Plot ──────────────────────────────────────────────────────────────────────
df_true <- data.frame(x = x_plot, y = y_plot)
df_obs  <- data.frame(x = x_obs,  y = y_obs)

ggplot() +
  geom_ribbon(data = df_true,
              aes(x = x, ymin = y - 2 * sigma_noise, ymax = y + 2 * sigma_noise),
              fill = "gray70", alpha = 0.3) +
  geom_line(data = df_true, aes(x = x, y = y), color = "black", linewidth = 0.8) +
  geom_point(data = df_obs,  aes(x = x, y = y), color = "steelblue", size = 0.8, alpha = 0.7) +
  labs(x = "x", y = "y", title = "True function and noisy observations") +
  theme_bw(base_size = 14)
