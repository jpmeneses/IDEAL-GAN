import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K

# ==============================================================================
# =                             Self-Attention Layer                           =
# ==============================================================================

class SelfAttention(Layer):
 
    def __init__(self, ch, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels
 
    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
 
        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)
 
        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True
 
    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
 
        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
 
        s = K.batch_dot(
            hw_flatten(g), K.permute_dimensions(
                hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]
 
        beta = K.softmax(s, axis=-1)  # attention map
 
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
 
        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x
 
        return x
 
    def compute_output_shape(self, input_shape):
        return input_shape

def AdaIN(content_features, style_features, alpha=1.0, epsilon=1e-5):
    '''
    Normalizes the `content_features` with scaling and offset from `style_features`.
    See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
    '''
    style_mean, style_variance = tf.nn.moments(style_features, [1], keepdims=True)
    content_mean, content_variance = tf.nn.moments(content_features, [1,2], keepdims=True)
    normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                            content_variance, style_mean, 
                                                            tf.sqrt(style_variance), epsilon)
    normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
    return normalized_content_features

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


def ResnetGenerator(input_shape=(256, 256, 3),
                    output_channels=3,
                    dim=64,
                    n_downsamplings=3,
                    n_blocks=9,
                    norm='instance_norm'):
    Norm = _get_norm_layer(norm)

    def _residual_block(x):
        dim = x.shape[-1]
        h = x

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        h = keras.layers.Conv2D(dim, 3, padding='valid', use_bias=False)(h)
        h = Norm()(h)

        return keras.layers.add([x, h])

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(dim, 7, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.nn.relu(h)

    # 2
    for _ in range(n_downsamplings):
        dim *= 2
        h = keras.layers.Conv2D(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 3
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 4
    for _ in range(n_downsamplings):
        dim //= 2
        h = keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)

    # 5
    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = keras.layers.Conv2D(output_channels, 7, padding='valid')(h)
    h = tf.tanh(h)

    return keras.Model(inputs=inputs, outputs=h)

def PatchGAN(input_shape,
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

    # 1
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, in_kernel, strides=2, padding='same', kernel_initializer='he_normal'))
    h = conv2d(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 16)
        conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal'))
        h = conv2d(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 16)
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'))
    h = conv2d(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # Self-attention
    if self_attention:
        h = SelfAttention(ch=dim)(h)

    # 3
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(1, n_kernel, strides=1, padding='same', kernel_initializer='glorot_normal'))
    h = conv2d(h)

    return keras.Model(inputs=inputs, outputs=h)

def PatchGAN_vConv1(input_shape,
                    dim=64,
                    n_layers=4,
                    self_attention=True,
                    norm='instance_norm'):
    dim_ = dim
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = keras.Input(shape=input_shape)

    # 1
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, 1, strides=1, padding='same', kernel_initializer='he_normal'))
    h = conv2d(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_layers):
        dim = min(dim * 2, dim_ * 16)
        conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, 1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'))
        h = conv2d(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # Self-attention
    if self_attention:
        h = SelfAttention(ch=dim)(h)

    # 3
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(1, 1, strides=1, padding='same', kernel_initializer='glorot_normal'))
    h = conv2d(h)

    return keras.Model(inputs=inputs, outputs=h)

def ConvDiscriminator(input_shape,
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

    # 1
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, in_kernel, strides=2, padding='same', kernel_initializer='he_normal'))
    h = conv2d(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    for _ in range(n_downsamplings - 1):
        dim = min(dim * 2, dim_ * 16)
        conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=2, padding='same', use_bias=False, kernel_initializer='he_normal'))
        h = conv2d(h)
        h = Norm()(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)

    # 2
    dim = min(dim * 2, dim_ * 16)
    conv2d = tfa.layers.SpectralNormalization(keras.layers.Conv2D(dim, n_kernel, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal'))
    h = conv2d(h)
    h = Norm()(h)
    h = tf.nn.leaky_relu(h, alpha=0.2)

    # Self-attention
    if self_attention:
        h = SelfAttention(ch=dim)(h)

    # 3
    h = tf.keras.layers.Flatten()(h)
    dense = tfa.layers.SpectralNormalization(keras.layers.Dense(1, use_bias=False, kernel_initializer='glorot_normal'))
    h = dense(h)

    return keras.Model(inputs=inputs, outputs=h)

# ==============================================================================
# =                                 MDWF-Net                                   =
# ==============================================================================

def UNet_Generator(
    input_shape,
    num_outputs=6,
    filters=32,
    num_layers=4,
    norm='instance_norm'):
    
    Norm = _get_norm_layer(norm)

    def _upsample(filters, kernel_size, strides, padding):
        return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def _conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    ):
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(inputs)
        c = Norm()(c)
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(c)
        c = Norm()(c)
        return c

    x = inputs = keras.Input(input_shape)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            )
        down_layers.append(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        filters = filters * 2  # double the number of filters with each layer

    x = _conv2d_block(
        inputs=x,
        filters=filters,
        )

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = _upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        
        # Water/Fat decoder
        x = keras.layers.concatenate([x, conv])
        x = _conv2d_block(
            inputs=x,
            filters=filters
            )

    x = keras.layers.Conv2D(num_outputs, (1, 1), activation='tanh')(x)

    return keras.Model(inputs=inputs, outputs=x)

def MDWF_Generator(
    input_shape,
    te_input=False,
    te_shape=(6,),
    num_outputs=6,
    filters=72,
    num_layers=4,
    self_attention=False,
    norm='instance_norm'):
    
    Norm = _get_norm_layer(norm)

    def _upsample(filters, kernel_size, strides, padding):
        return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def _conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    ):
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(inputs)
        c = Norm()(c)
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(c)
        c = Norm()(c)
        return c

    x = inputs1 = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
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
        )

    # if te_input:
    #     # Fully-connected network for processing the vector with echo-times
    #     hgt_dim = input_shape[0] // (2**(len(down_layers)))
    #     wdt_dim = input_shape[1] // (2**(len(down_layers)))
    #     y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_normal')(te)
    #     y = keras.layers.RepeatVector(hgt_dim*wdt_dim)(y)
    #     y = keras.layers.Reshape((hgt_dim,wdt_dim,filters))(y)

    #     # Add fully-connected output with latent space
    #     x = keras.layers.add([x,y])

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
        if self_attention and cont == 0:
            x2 = SelfAttention(ch=2*filters)(x2)
        x2 = _conv2d_block(
            inputs=x2,
            filters=filters
            )

        # R2* decoder
        x3 = keras.layers.concatenate([x3, conv])
        if self_attention and cont == 0:
            x3 = SelfAttention(ch=2*filters)(x3)
        x3 = _conv2d_block(
            inputs=x3,
            filters=filters
            )

        # Field map decoder
        x4 = keras.layers.concatenate([x4, conv])
        if cont == 0: # and self_attention
            x4 = SelfAttention(ch=2*filters)(x4)
        x4 = _conv2d_block(
            inputs=x4,
            filters=filters
            )

        # Update counter
        cont += 1

    x2 = keras.layers.Conv2D(4, (1, 1), activation='tanh', kernel_initializer='glorot_normal')(x2)
    x3 = keras.layers.Conv2D(1, (1, 1), activation='relu', kernel_initializer='he_normal')(x3)
    x4 = keras.layers.Conv2D(1, (1, 1), activation='tanh', kernel_initializer='glorot_normal')(x4)

    outputs = keras.layers.concatenate([x2,x3,x4])

    if te_input:
        return keras.Model(inputs=[inputs1,inputs2], outputs=outputs)
    else:
        return keras.Model(inputs=inputs1, outputs=outputs)

def PM_Generator(
    input_shape,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    R2_self_attention=False,
    FM_self_attention=True,
    norm='instance_norm'):
    
    Norm = _get_norm_layer(norm)

    def _upsample(filters, kernel_size, strides, padding):
        return keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def _conv2d_block(
        inputs,
        filters=16,
        kernel_size=(3, 3),
        kernel_initializer="he_normal",
        padding="same",
    ):
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(inputs)
        # c = tf.nn.relu6(c)
        c = Norm()(c)
        c = keras.layers.Conv2D(
            filters,
            kernel_size,
            activation='relu',
            kernel_initializer=kernel_initializer,
            padding=padding,
            use_bias=False,
            )(c)
        # c = tf.nn.relu6(c)
        c = Norm()(c)
        return c

    x = inputs = keras.Input(input_shape)
    if te_input:
        te = inputs2 = keras.Input(te_shape)

    down_layers = []
    for l in range(num_layers):
        x = _conv2d_block(
            inputs=x,
            filters=filters,
            )

        if te_input:
            # Fully-connected network for processing the vector with echo-times
            y = keras.layers.Dense(filters,activation='relu',kernel_initializer='he_uniform')(te)
            # Adaptive Instance Normalization for Style-Trasnfer
            x = AdaIN(x, y)

        down_layers.append(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        
        filters = filters * 2  # double the number of filters with each layer

    x = _conv2d_block(
        inputs=x,
        filters=filters,
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
            filters=filters
            )

        # Field map decoder
        x3 = keras.layers.concatenate([x3, conv])
        if FM_self_attention and cont == 0: 
            x3 = SelfAttention(ch=2*filters)(x3)
        x3 = _conv2d_block(
            inputs=x3,
            filters=filters
            )

        # Update counter
        cont += 1

    x2 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='glorot_normal')(x2)
    # x2 = tf.nn.relu6(x2)
    x3 = keras.layers.Conv2D(1, (1, 1), activation='tanh', kernel_initializer='glorot_normal')(x3)

    outputs = keras.layers.concatenate([x2,x3])

    if te_input:
        return keras.Model(inputs=[inputs,inputs2], outputs=outputs)
    else:
        return keras.Model(inputs=inputs, outputs=outputs)

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
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate * (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
