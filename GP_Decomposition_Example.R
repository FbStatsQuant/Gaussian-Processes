library(ggplot2)

# --- Function and data ---
f <- function(x) 3*x + 1.5 * sin(6*pi*x) * exp(-4*x)

set.seed(42)
n      <- 500
x_obs  <- (1:n) / n
y_obs  <- f(x_obs) + rnorm(n, 0, 0.3)

x_plot <- seq(0, 1, length.out = 1000)
y_plot <- f(x_plot)

df_true <- data.frame(x = x_plot, y = y_plot)
df_obs  <- data.frame(x = x_obs,  y = y_obs)

# --- Plot ---
ggplot() +
  geom_point(data = df_obs,  aes(x = x, y = y),
             color = "red", size = 0.8, alpha = 0.5) +
  geom_line(data = df_true, aes(x = x, y = y),
            color = "black", linewidth = 0.9) +
  labs(x = "x", y = "y",
       title = "True function and noisy observations") +
  theme_bw(base_size = 14)

  # --- Kernels ---
k_linear <- function(x1, x2, sigma_b = 1) {
  sigma_b^2 * outer(x1, x2, "*")
}

k_se <- function(x1, x2, sigma_f = 1, ell = 0.1) {
  d <- outer(x1, x2, "-")
  sigma_f^2 * exp(-d^2 / (2 * ell^2))
}

# --- Gram matrices and alpha* ---
sigma_noise <- 0.3

K_lin_XX <- k_linear(x_obs, x_obs)
K_se_XX  <- k_se(x_obs, x_obs)
K_XX     <- K_lin_XX + K_se_XX

C        <- K_XX + sigma_noise^2 * diag(n)
alpha    <- solve(C, y_obs)

# --- Posterior mean decomposition at test points ---
K_lin_Xx <- k_linear(x_plot, x_obs)
K_se_Xx  <- k_se(x_plot, x_obs)

mu_linear <- K_lin_Xx %*% alpha
mu_se     <- K_se_Xx  %*% alpha
mu_full   <- mu_linear + mu_se

# --- Plots ---
df_decomp <- data.frame(
  x          = x_plot,
  mu_full    = mu_full,
  mu_linear  = mu_linear,
  mu_se      = mu_se,
  f_true     = y_plot,
  f_linear   = 3 * x_plot,
  f_se       = 1.5 * sin(6 * pi * x_plot) * exp(-4 * x_plot)
)

# --- Plots ---
df_decomp <- data.frame(
  x          = x_plot,
  mu_full    = mu_full,
  mu_linear  = mu_linear,
  mu_se      = mu_se,
  f_true     = y_plot,
  f_linear   = 3 * x_plot,
  f_se       = 1.5 * sin(6 * pi * x_plot) * exp(-4 * x_plot)
)

# Full fit vs truth
ggplot(df_decomp) +
  geom_point(data = df_obs, aes(x = x, y = y), color = "red", size = 0.8, alpha = 0.4) +
  geom_line(aes(x = x, y = f_true),  color = "black", linewidth = 0.9) +
  geom_line(aes(x = x, y = mu_full), color = "blue",  linewidth = 0.9, linetype = "dashed") +
  labs(title = "Full fit: f_hat vs f", x = "x", y = "y") +
  theme_bw(base_size = 14)

# Linear component
ggplot(df_decomp) +
  geom_line(aes(x = x, y = f_linear),  color = "black", linewidth = 0.9) +
  geom_line(aes(x = x, y = mu_linear), color = "blue",  linewidth = 0.9, linetype = "dashed") +
  labs(title = "Linear component: mu_linear vs 3x", x = "x", y = "y") +
  theme_bw(base_size = 14)

# SE component
ggplot(df_decomp) +
  geom_line(aes(x = x, y = f_se),  color = "black", linewidth = 0.9) +
  geom_line(aes(x = x, y = mu_se), color = "blue",  linewidth = 0.9, linetype = "dashed") +
  labs(title = "SE component: mu_SE vs damped oscillation", x = "x", y = "y") +
  theme_bw(base_size = 14)