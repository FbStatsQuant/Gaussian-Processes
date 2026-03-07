library(kernlab)
library(DiceKriging)
library(GPfit)
library(mlegp)
library(ggplot2)
library(dplyr)

# true function: slow sinusoid + fast decaying oscillation
f <- function(x) sin(x) * 2*exp(-x / 5) + 0.5 * sin(5 * x)

sigma_noise <- 0.4
set.seed(42)

x_plot <- seq(0, 10, length.out = 500)
y_plot <- f(x_plot)

n_obs <- 200
x_obs <- sort(runif(n_obs, 0, 10))
y_obs <- f(x_obs) + rnorm(n_obs, 0, sigma_noise)

df_true <- data.frame(x = x_plot, y = y_plot)
df_obs  <- data.frame(x = x_obs,  y = y_obs)

windows(width = 20, height = 10)
ggplot() +
  geom_ribbon(data = df_true,
              aes(x = x, ymin = y - 2 * sigma_noise, ymax = y + 2 * sigma_noise),
              fill = "gray", alpha = 0.4) +
  geom_line(data = df_true, aes(x = x, y = y), color = "black", linewidth = 0.8) +
  geom_point(data = df_obs,  aes(x = x, y = y), color = "red", size = 0.8, alpha = 0.6) +
  labs(x = "x", y = "y", title = "True function and noisy observations") +
  theme_bw(base_size = 14)

# fit GP with Matern 5/2 kernel, noise estimated from data
gp_fit <- km(
  formula   = ~1,
  design    = data.frame(x = x_obs),
  response  = y_obs,
  covtype   = "matern5_2",
  nugget.estim = TRUE
)

# predict on dense grid
gp_pred <- predict(gp_fit, newdata = data.frame(x = x_plot), type = "UK")

df_gp <- data.frame(
  x    = x_plot,
  mean = gp_pred$mean,
  lo   = gp_pred$mean - 2 * gp_pred$sd,
  hi   = gp_pred$mean + 2 * gp_pred$sd
)

mae <- mean(abs(df_gp$mean - y_plot))
mse <- mean((df_gp$mean - y_plot)^2)

windows(width = 20, height = 10)
ggplot() +
  geom_ribbon(data = df_gp, aes(x = x, ymin = lo, ymax = hi),
              fill = "blue", alpha = 0.15) +
  geom_line(data = df_true, aes(x = x, y = y), color = "black", linewidth = 0.8) +
  geom_line(data = df_gp,   aes(x = x, y = mean), color = "blue", linewidth = 0.8) +
  geom_point(data = df_obs, aes(x = x, y = y), color = "red", size = 0.8, alpha = 0.6) +
  annotate("text", x = -Inf, y = Inf,
           label = sprintf("MAE = %.4f\nMSE = %.4f", mae, mse),
           hjust = -0.1, vjust = 1.5, size = 5) +
  labs(x = "x", y = "y", title = "GP fit") +
  theme_bw(base_size = 14)
