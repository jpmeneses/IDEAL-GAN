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
        std_map = tf.math.sqrt(var_map)
        log_std = tf.where(std_map!=0.0, tf.math.log(std_map), 0.0)
        msd = tf.square(y_true - y_pred)
        STDw_msd = tf.math.divide_no_nan(msd, std_map)
        return tf.reduce_mean(STDw_msd + log_std)


class VarMeanSquaredErrorR2(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        idx = y_pred.shape[-1]//2
        var_map = y_pred[...,idx:]
        y_pred = y_pred[...,:idx]
        # Based on ISMRM 2024 abstract No 1766: Non-central chi likelihood loss for 
        # quantitative MRI from parallel acquisitions with self-supervised deep learning
        loglik = tf.where(y_true>0.0,tf.math.log(y_true),0.0)
        loglik -= tf.where(var_map>0.0,tf.math.log(var_map),0.0)
        loglik -= tf.math.divide_no_nan(tf.square(y_true)+tf.square(y_pred),2*var_map)
        aux_log = tf.math.bessel_i0e(tf.math.divide_no_nan(y_true*y_pred,var_map))
        loglik += tf.where(aux_log>0.0,tf.math.log(aux_log),0.0)
        loglik += tf.math.divide_no_nan(y_true*y_pred,var_map)
        return tf.reduce_mean(-loglik)


class AbsolutePhaseDisparity(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_real = y_true[:,:1,:,:,:] * tf.math.cos(y_true[:,1:,:,:,:]*np.pi)
        y_true_imag = y_true[:,:1,:,:,:] * tf.math.sin(y_true[:,1:,:,:,:]*np.pi)
        y_pred_real = y_pred[:,:1,:,:,:] * tf.math.cos(y_pred[:,1:,:,:,:]*np.pi)
        y_pred_imag = y_pred[:,:1,:,:,:] * tf.math.sin(y_pred[:,1:,:,:,:]*np.pi)
        # Considering: (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
        y_prod_conj_real = (y_true_real*y_pred_real + y_true_imag*y_pred_imag)
        y_prod_conj_imag = (-y_true_real*y_pred_imag + y_true_imag*y_pred_real)
        y_prod_conj_phase = tf.math.atan2(y_prod_conj_imag, y_prod_conj_real)
        y_APD_num = y_true[:,:1,:,:,:]*tf.abs(y_prod_conj_phase)
        y_APD_num_sum = tf.reduce_sum(y_APD_num, axis=(2,3,4))
        y_APD_den_sum = tf.reduce_sum(y_true[:,:1,:,:,:], axis=(2,3,4))
        return tf.math.divide_no_nan(y_APD_num_sum, y_APD_den_sum)