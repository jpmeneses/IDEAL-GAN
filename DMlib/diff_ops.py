import numpy as np
import tensorflow as tf

import DLlib as dl

def set_key(key):
    np.random.seed(key)

def forward_noise(key, x_0, t, alpha_bar):
    set_key(key)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)
    noise = np.random.normal(size=x_0.shape)
    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise

def generate_timestamp(key, num, timesteps):
    set_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=timesteps, dtype=tf.int32)

# x_t = sqrt(alpha_t)*x_{t-1} + epsilon * (1 - alpha_t)
# x_{t-1} = (x_t - epsilon * (1 - alpha_t) ) / sqrt(alpha_t)
def ddpm(x_t, pred_noise, t, alpha, alpha_bar, beta):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)

    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5
    mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

    var = np.take(beta, t)
    z = np.random.normal(size=x_t.shape)

    return mean + (var ** .5) * z

def ddpm_add_cond(x_t, condition, t, alpha, alpha_bar, L1_norm=True, L1_w=1e-2):
    alpha_t = np.take(alpha, t)
    alpha_t_bar = np.take(alpha_bar, t)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    grad = dl.grad_xi(condition, x_t)
    eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5 
    res = eps_coef * grad * reshaped_one_minus_sqrt_alpha_bar_t
    if L1_norm:
        res += tf.math.sign(x_t) * L1_w
    return res

def ddim(x_t, pred_noise, t, sigma_t, alpha_bar):
    alpha_t_bar = np.take(alpha_bar, t)
    alpha_t_minus_one = np.take(alpha, t-1)

    pred = (x_t - ((1 - alpha_t_bar) ** 0.5) * pred_noise)/ (alpha_t_bar ** 0.5)
    pred = (alpha_t_minus_one ** 0.5) * pred

    pred = pred + ((1 - alpha_t_minus_one - (sigma_t ** 2)) ** 0.5) * pred_noise
    eps_t = np.random.normal(size=x_t.shape)
    pred = pred+(sigma_t * eps_t)

    return pred