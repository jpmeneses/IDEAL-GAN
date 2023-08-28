import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

vgg = keras.applications.vgg19.VGG19()

def perceptual_metric(input_shape, layers=[2,5,8,13,18], pad=(16,16)):
    inputs = keras.Input(input_shape)
    x = keras.layers.Lambda(lambda x: tf.concat([x*0.5+0.5,tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True))],axis=-1))(inputs)
    x = keras.layers.Lambda(lambda x: 255.0*x)(x)
    x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(x)
    x = keras.layers.ZeroPadding2D(padding=pad)(x)
    x = keras.applications.vgg19.preprocess_input(x)
    output = list()
    for l in layers:
        metric_vgg = keras.Model(inputs=vgg.inputs, outputs=vgg.layers[l].output)
        x_l = metric_vgg(x)
        output.append(x_l)
    
    return keras.Model(inputs=inputs, outputs=output)


def get_features(input_shape, layers=[2,5,8,13,18], pad=(16,16)):
    inputs = keras.Input(input_shape)
    x = keras.layers.Lambda(lambda x: tf.concat([x*0.5+0.5,tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True))],axis=-1))(inputs)
    x = keras.layers.Lambda(lambda x: 255.0*x)(x)
    x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(x)
    x = keras.layers.ZeroPadding2D(padding=pad)(x)

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

    covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma_x,sigma_y))

    # Product might be almost singular
    if not tf.math.reduce_any(tf.math.is_inf(covmean)):
        print(f"FID calculation produces singular product; adding {epsilon} to diagonal of covariance estimates")
        offset = tf.eye(sigma_x.shape[0]) * epsilon # CHECK INDEX
        covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma_x + offset, sigma_y + offset))

    # Numerical error might give slight imaginary component
    # if torch.is_complex(covmean):
    #     if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0, dtype=torch.double), atol=1e-3):
    #         raise ValueError(f"Imaginary component {torch.max(torch.abs(covmean.imag))} too high.")
    #     covmean = covmean.real

    tr_covmean = tf.linalg.trace(covmean)
    return tf.tensordot(diff, diff) + tf.linalg.trace(sigma_x) + tf.linalg.trace(sigma_y) - 2 * tr_covmean


class FID(keras.metrics.Metric):
    def __init__(self, name='FID_metric', **kwargs):
        super(FID, self).__init__(name=name, **kwargs)
        self.frechet_dist = self.add_weight(name='FID_dist', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.double)
        y_pred = tf.cast(y_pred, tf.double)

        mu_y_pred = tf.reduce_mean(y_pred, axis=0)
        sigma_y_pred = tfp.stats.covariance(y_pred)
        mu_y_true = tf.reduce_mean(y_true, axis=0)
        sigma_y_true = tfp.stats.covariance(y_true)

        self.frechet_dist.assign_add(compute_frechet_distance(mu_y_pred, sigma_y_pred, mu_y_true, sigma_y_true))

    def result(self):
        return self.frechet_dist