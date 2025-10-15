import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import Loss

def get_gan_losses_fn():
    bce = tf.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(r_logit, f_logit):
        r_loss = bce(tf.ones_like(r_logit), r_logit)
        f_loss = bce(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = bce(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v1_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(tf.maximum(1 - f_logit, 0))
        return f_loss

    return d_loss_fn, g_loss_fn


def get_hinge_v2_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = tf.reduce_mean(tf.maximum(1 - r_logit, 0))
        f_loss = tf.reduce_mean(tf.maximum(1 + f_logit, 0))
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = tf.reduce_mean(- f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_lsgan_losses_fn():
    mse = tf.losses.MeanSquaredError()

    def d_loss_fn(r_logit, f_logit):
        r_loss = mse(tf.ones_like(r_logit), r_logit)
        f_loss = mse(tf.zeros_like(f_logit), f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = mse(tf.ones_like(f_logit), f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_wgan_losses_fn():
    def d_loss_fn(r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss

    def g_loss_fn(f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    return d_loss_fn, g_loss_fn


def get_adversarial_losses_fn(mode):
    if mode == 'gan':
        return get_gan_losses_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_losses_fn()
    elif mode == 'hinge_v2':
        return get_hinge_v2_losses_fn()
    elif mode == 'lsgan':
        return get_lsgan_losses_fn()
    elif mode == 'wgan':
        return get_wgan_losses_fn()


def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp

    if mode == 'none':
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)

    return gp


def R1_regularization(f, real_sample):
    with tf.GradientTape() as t:
        t.watch(real_sample)
        pred_real = f(real_sample)
    grad_real = tf.convert_to_tensor(t.gradient(pred_real,real_sample))
    norm_grad = tf.reduce_sum(tf.reshape(grad_real**2, [tf.shape(grad_real)[0], -1]), axis=1)
    reg_loss = tf.reduce_mean(norm_grad)
    return tf.cast(reg_loss, tf.float32)


class VarMeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        idx = y_pred.shape[-1]//2
        var_map = y_pred[...,idx:]
        y_pred = y_pred[...,:idx]
        var_map = tf.where(var_map>=1e-5, var_map, 1e-5)
        std_map = tf.math.sqrt(var_map)
        log_std = tf.math.log(std_map)
        msd = tf.square(y_true - y_pred)
        STDw_msd = tf.math.divide_no_nan(msd, std_map)
        return tf.reduce_mean(STDw_msd + log_std)


class VarMeanSquaredErrorR2(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        idx = y_pred.shape[-1]//2
        var_map = y_pred[...,idx:]
        y_pred = y_pred[...,:idx]
        var_map = tf.where(var_map>=1e-5, var_map, 1e-5)
        # Based on ISMRM 2024 abstract No 1766: Non-central chi likelihood loss for 
        # quantitative MRI from parallel acquisitions with self-supervised deep learning
        loglik = tf.where(y_true>1e-5,tf.math.log(y_true),0.0)
        loglik -= tf.math.log(var_map)
        loglik -= tf.math.divide_no_nan(tf.square(y_true)+tf.square(y_pred),2*var_map)
        prod_div_aux = tf.math.divide_no_nan(y_true*y_pred,var_map)
        prod_div_aux = tf.clip_by_value(prod_div_aux, -50, 50)
        aux_log = tf.math.bessel_i0e(prod_div_aux)
        loglik += tf.where(aux_log>0.0,tf.math.log(aux_log),0.0)
        loglik += tf.math.divide_no_nan(y_true*y_pred,var_map)
        return tf.reduce_mean(-loglik)


class AbsolutePhaseDisparity(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_real = y_true[...,:1] * tf.math.cos(y_true[...,1:]*np.pi)
        y_true_imag = y_true[...,:1] * tf.math.sin(y_true[...,1:]*np.pi)
        y_pred_real = y_pred[...,:1] * tf.math.cos(y_pred[...,1:]*np.pi)
        y_pred_imag = y_pred[...,:1] * tf.math.sin(y_pred[...,1:]*np.pi)
        # Considering: (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        y_prod_conj_real = (y_true_real*y_pred_real + y_true_imag*y_pred_imag)
        y_prod_conj_imag = (-y_true_real*y_pred_imag + y_true_imag*y_pred_real)
        y_prod_conj_phase = tf.math.atan2(y_prod_conj_imag, y_prod_conj_real)
        y_APD_num = y_true[...,:1]*tf.abs(y_prod_conj_phase)
        y_APD_num_sum = tf.reduce_sum(y_APD_num, axis=(1,2,3,4))
        y_APD_den_sum = tf.reduce_sum(y_true[...,:1], axis=(1,2,3,4))
        return tf.math.divide_no_nan(y_APD_num_sum, y_APD_den_sum)


class RicianNLL(tf.keras.losses.Loss):
    def call(y_true, rv_y):
        """
        y_true: ground-truth tensor (shape matches rv_y batch/event)
        rv_y: tfd.Distribution (custom Rician exposing .nu and .sigma tensors)
        returns: scalar loss per batch
        """
        # 1) negative log-likelihood (mean over batch & pixels)
        logp = rv_y.log_prob(y_true)                      # shape [batch, ...]
        nll = -tf.reduce_mean(logp)

        # 2) extract parameters (make sure your Rician class exposes them)
        nu = tf.cast(rv_y.nu, tf.float32)
        sigma = tf.cast(rv_y.sigma, tf.float32)

        # Use a stable sigma (parametrization should already use softplus)
        sigma_safe = tf.maximum(sigma, sigma_min)

        # 3) soft penalty to encourage nu > sigma:
        # penalize where sigma > nu (i.e. max(0, sigma - nu))
        penalty_nu_gt_sigma = tf.reduce_mean(tf.nn.relu(sigma_safe - nu))

        # 4) penalty to avoid tiny sigma (push sigma above sigma_min)
        penalty_sigma_floor = tf.reduce_mean(tf.nn.relu(sigma_min - sigma_safe))

        # 5) optional L2 on (nu, log(sigma)) to avoid runaway values
        l2_term = tf.reduce_mean(tf.square(nu)) + tf.reduce_mean(tf.square(tf.math.log(sigma_safe + 1e-12)))

        loss = nll 

        return loss