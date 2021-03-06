import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

from DLlib import bn, complex_utils
from DLlib.attention import SelfAttention, AdaIN

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


# ==============================================================================
# =                                 MDWF-Net                                   =
# ==============================================================================

def MDWF_Generator(
    input_shape,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    WF_self_attention=False,
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
            filters=filters
            )

        # R2* decoder
        x3 = keras.layers.concatenate([x3, conv])
        if R2_self_attention and cont == 0:
            x3 = SelfAttention(ch=2*filters)(x3)
        x3 = _conv2d_block(
            inputs=x3,
            filters=filters
            )

        # Field map decoder
        x4 = keras.layers.concatenate([x4, conv])
        if FM_self_attention and cont == 0:
            x4 = SelfAttention(ch=2*filters)(x4)
        x4 = _conv2d_block(
            inputs=x4,
            filters=filters
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


def PM_complex(
    input_shape,
    te_input=False,
    te_shape=(6,),
    filters=72,
    num_layers=4,
    self_attention=False,
    norm='instance_norm'):

    def _conv2d_block(
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
        x = _conv2d_block(
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

    x = _conv2d_block(
        inputs=x,
        filters=filters,
        )

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        x = complex_utils.complex_Conv2DTranspose(filters, (2,2), (2,2))(x)

        x = keras.layers.concatenate([x, conv])
        # if self_attention and cont == 0:
        #     x = SelfAttention(ch=2*filters)(x)
        x = _conv2d_block(
            inputs=x,
            filters=filters
            )

    output = complex_utils.complex_Conv2D(1, (1, 1), kernel_initializer='glorot_normal', activation='cardioid')(x)

    if te_input:
        return keras.Model(inputs=[inputs,inputs2], outputs=output)
    else:
        return keras.Model(inputs=inputs, outputs=output)

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

