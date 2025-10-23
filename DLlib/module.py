import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow_probability as tfp

from DLlib import bn, complex_utils
from DLlib.attention import SelfAttention, AdaIN

# ==============================================================================
# =                       Rician distribution TFP layer                        =
# ==============================================================================

tfd = tfp.distributions
tfb = tfp.bijectors

def softplus_lb(x, ln_offset=1.0001):
    return tf.math.log(ln_offset + tf.math.exp(x))

class Rician(tfd.Distribution):
    def __init__(self, nu, sigma, validate_args=False, allow_nan_stats=True, name="Rician"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._nu = tf.convert_to_tensor(nu, name="nu")
            self._sigma = tf.convert_to_tensor(sigma, name="sigma")
            super(Rician, self).__init__(
                dtype=self._nu.dtype,
                reparameterization_type=tfd.NOT_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name
            )

    @property
    def nu(self):
        return self._nu

    @property
    def sigma(self):
        return self._sigma

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(tf.shape(self._nu), tf.shape(self._sigma))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self._nu.shape, self._sigma.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _log_prob(self, x):
        x = tf.convert_to_tensor(x, dtype=self.dtype)
        nu = self._nu
        sigma = self._sigma
        tf.debugging.assert_all_finite(x, "y contained NaN/Inf")
        tf.debugging.assert_all_finite(nu, "nu contained NaN/Inf")
        tf.debugging.assert_all_finite(sigma, "sigma contained NaN/Inf")

        # Compute argument of the Bessel function
        arg = tf.clip_by_value(x * nu / tf.square(sigma), -50.0, 50.0)
        tf.debugging.assert_all_finite(arg, "x contained NaN/Inf before bessel_i0")

        # Use exponentially scaled Bessel function for numerical stability:
        # log(I0(x)) = log(I0e(x)) + |x|
        log_bessel = tf.math.log(tf.math.bessel_i0e(arg)) + tf.abs(arg)
        tf.debugging.assert_all_finite(log_bessel, "bessel_i0e produced NaN/Inf")

        # Combine all terms
        log_unnorm = (
            tf.math.log(x) - 2.0 * tf.math.log(sigma)
            - (x**2 + nu**2) / (2.0 * sigma**2)
        )
        tf.debugging.assert_all_finite(log_unnorm, "log unnorm produced NaN/Inf")
        return log_unnorm + log_bessel

    def _sample_n(self, n, seed=None):
        # Sampling: Rician(nu, sigma) = sqrt((X + nu)^2 + Y^2), 
        # where X,Y ~ N(0, sigma^2) iid
        shape = tf.constant([n])
        normal = tfd.Normal(loc=0.0, scale=self._sigma)
        x = normal.sample(shape, seed=seed)
        y = normal.sample(shape, seed=seed)
        return tf.sqrt((x + self._nu)**2 + y**2)

    def _mean(self):
        nu = self._nu
        sigma = self._sigma

        x = -tf.square(nu) / (2.0 * tf.square(sigma))
        half_x = -x / 2.0

        # Compute L_{1/2}(x) = exp(x/2) * [ (1 - x) I0(-x/2) - x I1(-x/2) ]
        log_exp_term = x / 2.0 # + tf.abs(half_x)
        log_L = log_exp_term + tf.math.log(
            (1.0 - x) * tf.math.bessel_i0e(half_x) - x * tf.math.bessel_i1e(half_x) + 1e-12
        )
        tf.debugging.assert_all_finite(log_L, "log_L contained NaN/Inf after bessel_i0e [mean calc]")
        L = tf.exp(log_L)
        tf.debugging.assert_all_finite(L, "L contained NaN/Inf after exp [mean calc]")

        return sigma * tf.sqrt(np.pi / 2.0) * L

    def _variance(self):
        nu = self._nu
        sigma = self._sigma

        x = -tf.square(nu) / (2.0 * tf.square(sigma))
        half_x = -x / 2.0

        # Compute L_{1/2}(x) = exp(x/2) * [ (1 - x) I0(-x/2) - x I1(-x/2) ]
        log_exp_term = x / 2.0 # + tf.abs(half_x)
        log_L = log_exp_term + tf.math.log(
            (1.0 - x) * tf.math.bessel_i0e(half_x) - x * tf.math.bessel_i1e(half_x) + 1e-12
        )
        tf.debugging.assert_all_finite(log_L, "log_L contained NaN/Inf after bessel_i0e [var calc]")
        L = tf.exp(log_L)
        tf.debugging.assert_all_finite(L, "L contained NaN/Inf after exp [var calc]")

        return (
            2.0 * tf.square(sigma)
            + tf.square(nu)
            - (np.pi * tf.square(sigma) / 2.0) * tf.square(L)
        )


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def _upsample(filters, kernel_size, strides, padding, method='Conv2DTranspose'):
    if method == 'Conv2DTranspose':
        op = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    elif method == "Interpol_Conv":
        op = keras.Sequential()
        op.add(keras.layers.UpSampling2D(size=(2,2),interpolation='nearest'))
        op.add(keras.layers.Conv2D(filters, kernel_size, strides=1, padding=padding))
    return op


def _conv2d_block(
    inputs,
    filters,
    dropout=0.0,
    downsampling=False,
    kernel_size=(3, 3),
    kernel_initializer="he_normal",
    activation='relu',
    padding="same",
    norm="instance_norm"
):
    Norm = _get_norm_layer(norm)
    if downsampling:
        last_stride=2
    else:
        last_stride=1
    c = keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=False,
        )(inputs)
    c = Norm()(c)
    if dropout > 0.0:
        c = keras.layers.SpatialDropout2D(dropout)(c)
    c = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=last_stride,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=False,
        )(c)
    c = Norm()(c)
    return c


def _residual_block(x, norm, groups=1, Bayes=False):
    Norm = _get_norm_layer(norm)
    dim = x.shape[-1]
    h = x

    if Bayes:
        h = tfp.layers.Convolution2DFlipout(dim, 3, padding='same')(h)
    else:
        h = keras.layers.Conv2D(dim, 3, groups=groups, kernel_initializer='he_normal', padding='same', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h)

    if Bayes:
        h = tfp.layers.Convolution2DFlipout(dim, 3, padding='same')(h)
    else:
        h = keras.layers.Conv2D(dim, 3, groups=groups, kernel_initializer='he_normal', padding='same', use_bias=False)(h)
    h = Norm()(h)

    return keras.layers.add([x, h])


class FourierLayer(tf.keras.layers.Layer):
    def __init__(self, multi_echo=True):
        super(FourierLayer, self).__init__()
        self.multi_echo = multi_echo

    def call(self, x, training=None):
        # Generate complex x
        if self.multi_echo:
            ini_shape = x.shape
            x = keras.layers.Lambda(lambda z: tf.reshape(z,[-1,z.shape[2],z.shape[3],z.shape[4]]))(x)
            
        real_x = x[:,:,:,0]
        imag_x = x[:,:,:,1]
        x_complex = tf.complex(real_x,imag_x)

        x_fourier = tf.signal.fftshift(tf.signal.fft2d(x_complex),axes=(1,2))
        
        # Split into real and imaginary channels
        Re_gt = tf.math.real(tf.expand_dims(x_fourier,-1))
        Im_gt = tf.math.imag(tf.expand_dims(x_fourier,-1))
        res_gt = tf.concat([Re_gt,Im_gt],axis=-1)

        if self.multi_echo:
            res_gt = keras.layers.Lambda(lambda z: tf.reshape(z,ini_shape))(res_gt)

        return res_gt


def CriticZ(input_shape,
            n_downsamplings=3,
            dim=64,
            kernel=3,
            self_attention=True,
            ):
    h = inputs = keras.Input(shape=input_shape)
    for n in range(n_downsamplings):
        if self_attention:
            h = SelfAttention(ch=h.shape[-1])(h)
        h = keras.layers.Conv2D(dim, kernel, padding='same', strides=2, activation=tf.nn.leaky_relu, kernel_initializer='he_normal')(h)
        dim //= 4
    return keras.Model(inputs=inputs, outputs=h)


def PatchGAN(input_shape,
            cGAN=False,
            multi_echo=False,
            n_groups=1,
            dim=64,
            n_downsamplings=3,
            in_kernel=4,
            n_kernel=4,
            self_attention=True,
            norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)
    if cGAN:
        h2 = inputs2 = keras.Input(shape=input_shape)
        h = keras.layers.concatenate([h, h2])
    if multi_echo:
        h = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(h)

    # 1
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, in_kernel, strides=2, padding='same', groups=n_groups, kernel_initializer='he_normal'))
    h = conv2d(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 16)
        conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=2, padding='same', groups=n_groups, use_bias=False, kernel_initializer='he_normal'))
        h = conv2d(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 16)
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=1, padding='same', groups=n_groups, use_bias=False, kernel_initializer='he_normal'))
    h = conv2d(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # Self-attention
    if self_attention:
        h = SelfAttention(ch=dim)(h)

    # 3
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(1, n_kernel, strides=1, padding='same', kernel_initializer='glorot_normal'))
    h = conv2d(h)

    if cGAN:
        return keras.Model(inputs=[inputs, inputs2], outputs=h)
    else:
        return keras.Model(inputs=inputs, outputs=h)


def sGAN(input_shape,
        gen_mode=False,
        num_filters=32,
        num_layers=5,
        kernel_size=3
        ):
    x = inputs = keras.Input(shape=input_shape)
    h = list()
    for _ in range(num_layers):
        x = keras.layers.Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x)
        h.append(x)
    if gen_mode:
        x = keras.layers.Conv2D(input_shape[-1], 3, padding='same')(x)
        output = keras.layers.add([inputs,x])
        return keras.Model(inputs=inputs, outputs=output)
    else:
        return keras.Model(inputs=inputs, outputs=h)


# ==============================================================================
# =                               Custom CNNs                                  =
# ==============================================================================

def UNet(
    input_shape,
    n_out=1,
    skip_con=True,
    bayesian=False,
    std_out=False,
    ME_layer=False,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    dropout=0.0,
    output_activation='tanh',
    output_initializer='glorot_normal',
    self_attention=False,
    norm='instance_norm'):

    x = inputs1 = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    if ME_layer:
        x = keras.layers.ConvLSTM2D(filters,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    elif len(input_shape) > 3:
        x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[-3],x.shape[-2],x.shape[-1]]))(x)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        if te_input:
            # Fully-connected network for processing the vector with echo-times
            y = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(te)
            y = keras.layers.RNN(keras.layers.LSTMCell(6))(y)
            y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(y)
            # Adaptive Instance Normalization for Style-Transfer
            x = AdaIN(x, y)
        
        down_layers.append(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        filters = filters * 2  # double the number of filters with each layer

    x = _conv2d_block(
        inputs=x,
        filters=filters,
        dropout=dropout,
        norm=norm
        )

    cont = 0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)

        # Water/Fat decoder
        if skip_con:
            x = keras.layers.concatenate([x, conv])
        if self_attention and cont == 0:
            if skip_con:
                x = SelfAttention(ch=2*filters)(x)
            else:
                x = SelfAttention(ch=filters)(x)
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # Update counter
        cont += 1

    output = keras.layers.Conv2D(n_out, (1, 1), activation=output_activation, kernel_initializer=output_initializer)(x)
    if ME_layer:
        output = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(output)
    if bayesian or std_out:
        x_std = keras.layers.Conv2D(16, (1,1), activation='relu', kernel_initializer='he_uniform')(x)
        # Compute standard deviation (sigma; NOT sigma^2)
        out_var = keras.layers.Conv2D(n_out, (1,1), activation='sigmoid', kernel_initializer='he_normal')(x_std)
        if ME_layer:
            out_var = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(out_var)
        if bayesian:
            x_prob = keras.layers.concatenate([output,out_var])
            if output_activation == 'tanh':
                output = tfp.layers.DistributionLambda(
                            lambda t: tfp.distributions.Normal(
                                loc=t[...,:n_out],
                                scale=t[...,n_out:])
                            )(x_prob)
            else:
                # Based on: https://en.wikipedia.org/wiki/Folded_normal_distribution#Related_distributions
                output = tfp.layers.DistributionLambda(
                            lambda t: Rician(
                                nu=t[...,:n_out],
                                sigma=t[...,n_out:])
                            )(x_prob)

    if te_input:
        return keras.Model(inputs=[inputs1,inputs2], outputs=output)
    elif std_out:
        return keras.Model(inputs=inputs1, outputs=[output,out_var])
    else:
        return keras.Model(inputs=inputs1, outputs=output)


def CSE_sample(input_shape):
    mean = inputs1 = keras.Input(input_shape)
    std = inputs2 = keras.Input(input_shape)
    x_prob = keras.layers.concatenate([mean, std])
    out_prob = tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.Normal(
                    loc=t[...,:input_shape[-1]],
                    scale=tf.math.sqrt(t[...,input_shape[-1]:])),
                )(x_prob)
    return keras.Model(inputs=[inputs1,inputs2], outputs=out_prob)


def MDWF_Generator(
    input_shape,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    dropout=0.0,
    WF_self_attention=False,
    R2_self_attention=False,
    FM_self_attention=True,
    norm='instance_norm'):
    
    x = inputs1 = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            dropout=dropout,
            norm=norm
            )
        down_layers.append(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        if te_input and l==1:
            # Fully-connected network for processing the vector with echo-times
            hgt_dim = input_shape[0] // (2**(l+1))
            wdt_dim = input_shape[1] // (2**(l+1))
            y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(te)
            y = keras.layers.RepeatVector(hgt_dim*wdt_dim)(y)
            y = keras.layers.Reshape((hgt_dim,wdt_dim,filters))(y)

            # Add fully-connected output with latent space
            x = keras.layers.add([x,y])

        filters = filters * 2  # double the number of filters with each layer

    x = _conv2d_block(
        inputs=x,
        filters=filters,
        dropout=dropout,
        norm=norm
        )

    cont = 0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        if cont < 1:
            x2 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x3 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x4 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        elif cont >= 1:
            x2 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x2)
            x3 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x3)
            x4 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x4)

        # Water/Fat decoder
        x2 = keras.layers.concatenate([x2, conv])
        if WF_self_attention and cont == 0:
            x2 = SelfAttention(ch=2*filters)(x2)
        x2 = _conv2d_block(
            inputs=x2,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # R2* decoder
        x3 = keras.layers.concatenate([x3, conv])
        if R2_self_attention and cont == 0:
            x3 = SelfAttention(ch=2*filters)(x3)
        x3 = _conv2d_block(
            inputs=x3,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # Field map decoder
        x4 = keras.layers.concatenate([x4, conv])
        if FM_self_attention and cont == 0:
            x4 = SelfAttention(ch=2*filters)(x4)
        x4 = _conv2d_block(
            inputs=x4,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # Update counter
        cont += 1

    x2 = keras.layers.Conv2D(2, (1, 1), activation='sigmoid', kernel_initializer='glorot_normal')(x2)
    x3 = keras.layers.Conv2D(1, (1, 1), activation='relu', kernel_initializer='he_normal')(x3)
    x4 = keras.layers.Conv2D(1, (1, 1), activation='tanh', kernel_initializer='glorot_normal')(x4)

    outputs = keras.layers.concatenate([x2,x3,x4])

    if te_input:
        return keras.Model(inputs=[inputs1,inputs2], outputs=outputs)
    else:
        return keras.Model(inputs=inputs1, outputs=outputs)


def PM_Generator(
    input_shape,
    n_out=1,
    ME_layer=True,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    dropout=0.0,
    R2_init='glorot_normal',
    FM_init='glorot_normal',
    R2_self_attention=False,
    FM_self_attention=True,
    norm='instance_norm'):
    
    x = inputs = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    if ME_layer:
        x = keras.layers.ConvLSTM2D(filters,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    elif len(input_shape) > 3 and te_input == True:
        x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[-3],x.shape[-2],x.shape[-1]]))(x)
        # # Fully-connected network for processing the vector with echo-times
        # y = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(te)
        # y = keras.layers.RNN(keras.layers.LSTMCell(6))(y)
        # y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(y)
        # # Adaptive Instance Normalization for Style-Transfer
        # x = AdaIN(x, y)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        if te_input: # and not(ME_layer):
            # Fully-connected network for processing the vector with echo-times
            y = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=-1))(te)
            y = keras.layers.RNN(keras.layers.LSTMCell(6))(y)
            y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(y)
            # Adaptive Instance Normalization for Style-Transfer
            x = AdaIN(x, y)

        down_layers.append(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        filters = filters * 2  # double the number of filters with each layer

    x = _conv2d_block(
        inputs=x,
        filters=filters,
        dropout=dropout,
        norm=norm
        )

    cont = 0
    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        if cont < 1:
            x2 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
            x3 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        elif cont >= 1:
            x2 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x2)
            x3 = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x3)

        # R2* decoder
        x2 = keras.layers.concatenate([x2, conv])
        if R2_self_attention and cont == 0:
            x2 = SelfAttention(ch=2*filters)(x2)
        x2 = _conv2d_block(
            inputs=x2,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # Field map decoder
        x3 = keras.layers.concatenate([x3, conv])
        if FM_self_attention and cont == 0: 
            x3 = SelfAttention(ch=2*filters)(x3)
        x3 = _conv2d_block(
            inputs=x3,
            filters=filters,
            dropout=dropout,
            norm=norm
            )

        # if te_input:
        #     # Fully-connected network for processing the vector with echo-times
        #     y2 = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(te)
        #     y3 = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(te)
        #     # Adaptive Instance Normalization for Style-Trasnfer
        #     x2 = AdaIN(x2, y2)
        #     x3 = AdaIN(x3, y3)

        # Update counter
        cont += 1

    x2 = keras.layers.Conv2D(n_out, (1, 1), activation='sigmoid', kernel_initializer=R2_init)(x2)
    x3 = keras.layers.Conv2D(n_out, (1, 1), activation='tanh', kernel_initializer=FM_init)(x3)
    
    if ME_layer:
        outputs = keras.layers.concatenate([x3,x2])
        outputs = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(outputs)
    else:
        outputs = keras.layers.concatenate([x2,x3])

    if te_input:
        return keras.Model(inputs=[inputs,inputs2], outputs=outputs)
    else:
        return keras.Model(inputs=inputs, outputs=outputs)


def PM_complex(
    input_shape,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    self_attention=False,
    norm='instance_norm'):

    def _complex_conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
    ):
        def _concat_complex(x):
            x_real = tf.math.real(x)
            x_imag = tf.math.imag(x)
            return tf.concat((x_real,x_imag),axis=-1)

        def _merge_complex(x):
            x_dim = x.shape[-1] // 2
            x_real = x[:,:,:,:x_dim]
            x_imag = x[:,:,:,x_dim:]
            return tf.complex(x_real,x_imag)

        c = complex_utils.complex_Conv2D(
            filters,
            kernel_size,
            activation='crelu',
            use_bias=False,
            kernel_initializer=kernel_initializer)(inputs)
        c = bn.ComplexBatchNormalization()(_concat_complex(c))
        c = _merge_complex(c)
        c = complex_utils.complex_Conv2D(
            filters,
            kernel_size,
            activation='crelu',
            stride=1,
            use_bias=False,
            kernel_initializer=kernel_initializer)(c)
        c = bn.ComplexBatchNormalization()(_concat_complex(c))
        c = _merge_complex(c)
        return c

    x = inputs = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    down_layers = []
    for l in range(num_layers):
        x = _complex_conv2d_block(
            inputs=x,
            filters=filters,
            )

        # if te_input:
            # Fully-connected network for processing the vector with echo-times
            # y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(te)
            # Adaptive Instance Normalization for Style-Trasnfer
            # x = AdaIN(x, y)

        down_layers.append(x) # UNCOMMENT WHEN CONV_TRANSPOSE IS READY
        x = complex_utils.complex_MaxPool2D((2, 2))(x)

        filters *= 2  # double the number of filters with each layer

    x = _complex_conv2d_block(
        inputs=x,
        filters=filters,
        )

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = complex_utils.complex_Conv2DTranspose(filters, (2,2), (2,2))(x)

        x = keras.layers.concatenate([x, conv])
        # if self_attention and cont == 0:
        #     x = SelfAttention(ch=2*filters)(x)
        x = _complex_conv2d_block(
            inputs=x,
            filters=filters
            )

    output = complex_utils.complex_Conv2D(1, (1, 1), kernel_initializer='glorot_normal', activation='cardioid')(x)

    if te_input:
        return keras.Model(inputs=[inputs,inputs2], outputs=output)
    else:
        return keras.Model(inputs=inputs, outputs=output)


def encoder(
    input_shape,
    encoded_dims,
    multi_echo=True,
    filters=36,
    num_layers=4,
    num_res_blocks=2,
    dropout=0.0,
    sd_out=True,
    ls_mean_activ='leaky_relu',
    ls_reg_weight=1.0,
    NL_self_attention=True,
    norm='instance_norm'):

    x = inputs1 = keras.Input(input_shape)

    if not(isinstance(filters, list)):
        filters = [filters*2**k for k in range(num_layers+1)]
    if multi_echo:
        x = keras.layers.ConvLSTM2D(filters[0],3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    x = keras.layers.Conv2D(filters[0],3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer="he_normal")(x)

    for l in range(num_layers):
        for n_res in range(num_res_blocks):
            x = _residual_block(x, norm=norm)

        # Double the number of filters and downsample
        x = keras.layers.Conv2D(filters[l+1],3,strides=2,padding="same",activation=tf.nn.leaky_relu,kernel_initializer="he_normal")(x)

    if NL_self_attention:
        x = _residual_block(x, norm=norm)
        x = SelfAttention(ch=filters[-1])(x)
        x = _residual_block(x, norm=norm)
    
    if ls_mean_activ == 'leaky_relu':
        ls_mean_activ = tf.nn.leaky_relu
    elif ls_mean_activ == 'None':
        ls_mean_activ = None
    x = keras.layers.Conv2D(encoded_dims,3,padding="same",activation=ls_mean_activ,kernel_initializer="he_normal")(x)
    _,ls_hgt,ls_wdt,ls_dims = x.shape

    if sd_out:
        x_mean = keras.layers.Conv2D(encoded_dims,1,padding="same",activation=ls_mean_activ,kernel_initializer="he_normal")(x)
        x_mean = keras.layers.Flatten()(x_mean)

        x_std = keras.layers.Conv2D(encoded_dims,1,padding="same",activation='relu',kernel_initializer="he_normal")(x)
        x_std = keras.layers.Flatten()(x_std)
        
        x = keras.layers.concatenate([x_mean,x_std],axis=-1)
    
        prior=tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros((ls_hgt,ls_wdt,ls_dims)), scale=1),
                                            reinterpreted_batch_ndims=3)
        output = tfp.layers.IndependentNormal([ls_hgt,ls_wdt,encoded_dims],
                    activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=ls_reg_weight))(x)
    else:
        output = keras.layers.Conv2D(encoded_dims,1,padding="same")(x)

    return keras.Model(inputs=inputs1, outputs=output)


def decoder(
    encoded_dims,
    output_shape,
    multi_echo=True,
    n_groups=1,
    filters=36,
    num_layers=4,
    num_res_blocks=2,
    dropout=0.0,
    output_activation='tanh',
    output_initializer='glorot_normal',
    bayes_layer=False,
    NL_self_attention=True,
    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    hgt,wdt,n_out = output_shape
    hls = hgt//(2**(num_layers))
    wls = wdt//(2**(num_layers))
    input_shape = (hls,wls,encoded_dims)
    if not(isinstance(filters, list)):
        filters = [filters*2**k for k in range(num_layers+1)]
    filters.reverse()
    
    x = inputs1 = keras.Input(input_shape)

    x = keras.layers.Conv2D(encoded_dims,3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x)
    x = keras.layers.Conv2D(filters[0],3,padding="same",activation=tf.nn.leaky_relu,kernel_initializer='he_normal')(x) # n_groups
    if NL_self_attention:
        x = _residual_block(x, norm=norm) # n_groups
        x = SelfAttention(ch=filters[0])(x)
        x = _residual_block(x, norm=norm) # n_groups
    for l in range(num_layers):
        x = _upsample(filters[l+1], (2, 2), strides=(2, 2), padding='same', method='Interpol_Conv')(x)
        for n_res in range(num_res_blocks):
            x = _residual_block(x, norm=norm, groups=n_groups)

    x = Norm()(x)
    if bayes_layer:
        x = keras.layers.Conv2D(filters[-1],3,padding="same",groups=n_groups,activation=output_activation,kernel_initializer=output_initializer)(x)
        x_r = keras.layers.Lambda(lambda z: z[...,:filters[-1]//2])(x)
        x_i = keras.layers.Lambda(lambda z: z[...,filters[-1]//2:])(x)
        x_r = tfp.layers.Convolution2DFlipout(1,3,padding='same',activation=output_activation)(x_r)
        x_i = tfp.layers.Convolution2DFlipout(1,3,padding='same',activation=output_activation)(x_i)
        output = keras.layers.concatenate([x_r,x_i])
    else:
        output = keras.layers.Conv2D(n_out,3,padding="same",groups=n_groups,activation=output_activation,kernel_initializer=output_initializer)(x)
    if multi_echo:
        output = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(output)

    return keras.Model(inputs=inputs1, outputs=output)


def Bayes_decoder(
    encoded_dims,
    output_2D_shape,
    filters=36,
    num_layers=4,
    num_res_blocks=2,
    dropout=0.0,
    output_activation=None,
    output_initializer='glorot_normal',
    NL_self_attention=True,
    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    hgt,wdt = output_2D_shape
    hls = hgt//(2**(num_layers))
    wls = wdt//(2**(num_layers))
    filt_ini = filters*(2**num_layers)
    input_shape = (hls,wls,encoded_dims)
    
    x = inputs1 = keras.Input(input_shape)

    x = tfp.layers.Convolution2DFlipout(encoded_dims,3,padding='same',activation=tf.nn.leaky_relu)(x)
    x_r = keras.layers.Lambda(lambda z: z[...,:encoded_dims//2])(x)
    x_i = keras.layers.Lambda(lambda z: z[...,encoded_dims//2:])(x)
    x_list_in = [x_r,x_i]
    x_list_out = list()
    for __x in x_list_in:
        filt_iter = filt_ini
        _x = tfp.layers.Convolution2DFlipout(filt_iter,3,padding='same',activation=tf.nn.leaky_relu)(__x)
        if NL_self_attention:
            _x = _residual_block(_x, norm=norm, Bayes=True)
            _x = SelfAttention(ch=filt_ini)(_x)
            _x = _residual_block(_x, norm=norm, Bayes=True)
        for cont in range(num_layers):
            filt_iter //= 2  # decreasing number of filters with each layer
            _x = _upsample(filt_iter, (2, 2), strides=(2, 2), padding='same', method='Interpol_Conv')(_x)
            for n_res in range(num_res_blocks):
                _x = _residual_block(_x, norm=norm, Bayes=True)
        _x = Norm()(_x)
        _x = tfp.layers.Convolution2DFlipout(1,3,padding='same',activation=output_activation)(_x)
        x_list_out.append(_x)
    x = keras.layers.concatenate(x_list_out)
    output = keras.layers.Lambda(lambda z: tf.expand_dims(z,axis=1))(x)

    return keras.Model(inputs=inputs1, outputs=output)


# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================

class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        if self._steps > self._step_decay:
            self.current_learning_rate.assign(tf.cond(
                step >= self._step_decay,
                true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
                false_fn=lambda: self._initial_learning_rate
            ))
            return self.current_learning_rate
        else:
            return self._initial_learning_rate


# ==============================================================================
# =                         Indexes of decoder layers                          =
# ==============================================================================

def PM_decoder_idxs(decod_idx,
                    num_decoders,
                    num_levels,
                    R2_self_attention=False,
                    FM_self_attention=True):
    cnst = 1 + num_decoders
    conv2d_layers = 4
    level_layers = conv2d_layers + 2  # Transpose Conv2D + skip-connection
    if num_decoders < 1:
        NameError('CNN architecture must have 2 or more decoders')
    decod_layers = cnst + (num_levels)*(level_layers*num_decoders)
    sa_idx = cnst + (num_levels-1)*(level_layers*num_decoders) + 2*(conv2d_layers) +1 #48
    if R2_self_attention:
        decod_layers += 1
    if FM_self_attention:
        decod_layers += 1
    idxs = list()
    for a in range(decod_layers):
        if (a!=0 and (a+(num_decoders-1))%num_decoders==(num_decoders-decod_idx) and (a+1)<sa_idx):
            idxs.append(a+1)
        elif FM_self_attention^R2_self_attention and a+1==sa_idx:
            if (FM_self_attention and decod_idx==2) or (R2_self_attention and decod_idx==1):
                idxs.append(a+1)
        elif (FM_self_attention^R2_self_attention) and ((a+1)%num_decoders==(num_decoders-decod_idx) and (a+1)>sa_idx):
            idxs.append(a+1)
    return idxs


