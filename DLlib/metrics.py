import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

vgg = keras.applications.vgg19.VGG19()

def perceptual_metric(input_shape, layers=[2,5,8,13,18], multi_echo=True, only_mag=False):
    x = inputs = keras.Input(input_shape)
    if multi_echo:
        x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(x)
    x = keras.layers.Lambda(lambda x: tf.image.resize(x,[224,224],method='lanczos5'))(x)
    # x = keras.layers.ZeroPadding2D(padding=16)(x)
    if only_mag:
        x = keras.layers.Lambda(lambda x: tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True)))(x)
        x = keras.layers.Lambda(lambda x: tf.concat([x,x,x],axis=-1))(x)
    else:
        x =keras.layers.Lambda(lambda x: tf.concat([x[...,:1]*0.5+0.5,
                                                    tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True)),
                                                    x[...,1:2]*0.5+0.5],axis=-1))(x)
    x = keras.layers.Lambda(lambda x: 255.0*x)(x)
    x = keras.applications.vgg19.preprocess_input(x)
    output = list()
    for l in layers:
        metric_vgg = keras.Model(inputs=vgg.inputs, outputs=vgg.layers[l].output)
        x_l = metric_vgg(x)
        output.append(x_l)

    return keras.Model(inputs=inputs, outputs=output)


def get_features(input_shape, layers=[2,5,8,13,18]):
    inputs = keras.Input(input_shape)
    x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(inputs)
    x = keras.layers.Lambda(lambda x: tf.image.resize(x,[224,224],method='lanczos5'))(x)
    x = keras.layers.Lambda(lambda x: tf.concat([x[...,:1]*0.5+0.5,
                                                tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True)),
                                                x[...,1:2]*0.5+0.5],axis=-1))(x)
    x = keras.layers.Lambda(lambda x: 255.0*x)(x)

    # Change order from 'RGB' to 'BGR'
    # Subtract mean used during training
    x = keras.applications.vgg19.preprocess_input(x)

    # Get model outputs
    features = vgg(x)

    return keras.Model(inputs=inputs, outputs=features)


# def _cov(input_data: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
#     """
#     Estimate a covariance matrix of the variables.

#     Args:
#         input_data: A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a variable,
#             and each column a single observation of all those variables.
#         rowvar: If rowvar is True (default), then each row represents a variable, with observations in the columns.
#             Otherwise, the relationship is transposed: each column represents a variable, while the rows contain
#             observations.
#     """
#     if input_data.dim() < 2:
#         input_data = input_data.view(1, -1)

#     if not rowvar and input_data.size(0) != 1:
#         input_data = input_data.t()

#     factor = 1.0 / (input_data.size(1) - 1)
#     input_data = input_data - torch.mean(input_data, dim=1, keepdim=True)
#     return factor * input_data.matmul(input_data.t()).squeeze()


def compute_frechet_distance(mu_x, sigma_x, mu_y, sigma_y, epsilon = 1e-6):
    """The Frechet distance between multivariate normal distributions."""
    diff = mu_x - mu_y
    aux_covmean = tf.linalg.matmul(sigma_x,sigma_y)

    # Product might be almost singular
    if not tf.math.reduce_any(tf.math.is_inf(aux_covmean)):
        print(f"FID calculation produces singular product; adding {epsilon} to diagonal of covariance estimates")
        offset = tf.eye(sigma_x.shape[0], dtype=tf.float32) * epsilon # CHECK INDEX
        aux_covmean = tf.linalg.matmul(sigma_x + offset, sigma_y + offset)

    covmean = tf.math.real(tf.linalg.sqrtm(tf.complex(aux_covmean,0.0)))
    tr_covmean = tf.linalg.trace(covmean)
    return tf.reduce_sum(tf.multiply(diff,diff)) + tf.linalg.trace(sigma_x) + tf.linalg.trace(sigma_y) - 2 * tr_covmean


class FID(keras.metrics.Metric):
    def __init__(self, name='FID_metric', **kwargs):
        super(FID, self).__init__(name=name, **kwargs)
        self.frechet_dist = self.add_weight(name='FID_dist', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mu_y_pred = tf.reduce_mean(y_pred, axis=0)
        sigma_y_pred = tfp.stats.covariance(y_pred)
        mu_y_true = tf.reduce_mean(y_true, axis=0)
        sigma_y_true = tfp.stats.covariance(y_true)

        self.frechet_dist.assign_add(compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y_true, sigma_y_true))

    def result(self):
        return self.frechet_dist


class MMD(keras.metrics.Metric):
    def __init__(self, beta=1.0, gamma=2.0, name='MMD_metric', **kwargs):
        super(MMD, self).__init__(name=name, **kwargs)
        self.mmd_dist = self.add_weight(name='MMD_dist', initializer='zeros')
        self.beta = beta
        self.gamma = gamma

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true = tf.reshape(y_true, [y_true.shape[0],-1])
        y_pred = tf.reshape(y_pred, [y_pred.shape[0],-1])

        y_true_y_true = tf.linalg.matmul(y_true, y_true, transpose_b=True)
        y_pred_y_pred = tf.linalg.matmul(y_pred, y_pred, transpose_b=True)
        y_pred_y_true = tf.linalg.matmul(y_pred, y_true, transpose_b=True)

        y_true_y_true = y_true_y_true / y_true.shape[1]
        y_pred_y_pred = y_pred_y_pred / y_true.shape[1]
        y_pred_y_true = y_pred_y_true / y_true.shape[1]

        self.mmd_dist.assign_add(self.beta*(tf.reduce_mean(y_true_y_true) + tf.reduce_mean(y_pred_y_pred)) - self.gamma * tf.reduce_mean(y_pred_y_true))

    def result(self):
        return self.mmd_dist


# class MS_SSIM(keras.metrics.Metric):
#     def __init__(self, spatial_dims, data_range=1.0, kernel_size=11, kernel_sigma=1.5,
#                     k1=0.01, k2=0.03, weights=(0.0448,0.2856,0.3001,0.2363,0.1333), name='MS_SSIM_metric', **kwargs):
#         self.MS_SSIM_dist = self.add_weight(name='MS_SSIM_dist', initializer='zeros')

#         self.spatial_dims = spatial_dims
#         self.data_range = data_range

#         self.kernel_size = kernel_size
#         self.kernel_sigma = kernel_sigma

#         self.k1 = k1
#         self.k2 = k2
#         self.weights = weights

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         # check if image have enough size for the number of downsamplings and the size of the kernel
#         weights_div = max(1, (len(self.weights) - 1)) ** 2
#         y_pred_spatial_dims = y_pred.shape[:-1]
#         for i in range(len(y_pred_spatial_dims)):
#             if y_pred_spatial_dims[i] // weights_div <= self.kernel_size[i] - 1:
#                 raise ValueError(
#                     f"For a given number of `weights` parameters {len(self.weights)} and kernel size "
#                     f"{self.kernel_size[i]}, the image height must be larger than "
#                     f"{(self.kernel_size[i] - 1) * weights_div}."
#                 )

#         weights = torch.tensor(self.weights, device=y_pred.device, dtype=torch.float)

#         multiscale_list = []
#         for _ in range(len(weights)):
#             ssim, cs = compute_ssim_and_cs(
#                 y_pred=y_pred,
#                 y=y,
#                 spatial_dims=self.spatial_dims,
#                 data_range=self.data_range,
#                 kernel_type=self.kernel_type,
#                 kernel_size=self.kernel_size,
#                 kernel_sigma=self.kernel_sigma,
#                 k1=self.k1,
#                 k2=self.k2,
#             )

#             cs_per_batch = cs.view(cs.shape[0], -1).mean(1)

#             multiscale_list.append(torch.relu(cs_per_batch))
#             y_pred = avg_pool(y_pred, kernel_size=2)
#             y = avg_pool(y, kernel_size=2)

#         ssim = ssim.view(ssim.shape[0], -1).mean(1)
#         multiscale_list[-1] = torch.relu(ssim)
#         multiscale_list = torch.stack(multiscale_list)

#         ms_ssim_value_full_image = torch.prod(multiscale_list ** weights.view(-1, 1), dim=0)

#         ms_ssim_per_batch: torch.Tensor = ms_ssim_value_full_image.view(ms_ssim_value_full_image.shape[0], -1).mean(
#             1, keepdim=True
#         )


class CoVar(tf.keras.layers.Layer):
    def __init__(self):
        super(CoVar, self).__init__()

    def call(self, x, training=None):
        x = keras.layers.Flatten()(x)
        x_mu = tf.reduce_mean(x,axis=0,keepdims=True)
        x_dif = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(x - x_mu)
        cov = keras.layers.Lambda(lambda a: tf.linalg.matmul(a,a,transpose_b=True))(x_dif)
        cov_res = keras.layers.Lambda(lambda z: tf.reduce_mean(z,axis=0,keepdims=True))(cov)
        return cov_res
