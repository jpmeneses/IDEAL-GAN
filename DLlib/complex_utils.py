from math import pi

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K

class complex_Conv2D(Layer):

    def __init__(self, filters, kernel_size, stride=1, padding='valid', data_format="channels_last", dilation_rate=(1, 1),
                activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                bias_constraint=None, **kwargs):
        super(complex_Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = stride
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        kernel_shape = (1, 1) + (input_shape[-1], self.filters)

        # Create a trainable weight variable for this layer:
        self.kernel_xR= self.add_weight(name='kernel_xR',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        constraint=self.kernel_constraint)
        self.kernel_xI= self.add_weight(name='kernel_xI',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        constraint=self.kernel_constraint)

        super(complex_Conv2D, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4)
        self.built = True

    def call(self, x):
        def split_complex(x):
            return (tf.math.real(x), tf.math.imag(x))

        x_real, x_imag = split_complex(x)

        tf_real_real = K.conv2d(x_real, 
                                kernel=self.kernel_xR, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)
        tf_imag_real = K.conv2d(x_imag,
                                kernel=self.kernel_xR, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)
        tf_real_imag = K.conv2d(x_real,
                                kernel=self.kernel_xI, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)
        tf_imag_imag = K.conv2d(x_imag,
                                kernel=self.kernel_xI, strides=self.strides, padding=self.padding,
                                data_format=self.data_format, dilation_rate=self.dilation_rate)

        real_out = tf_real_real - tf_imag_imag
        imag_out = tf_imag_real + tf_real_imag
        
        if self.activation == 'crelu':
            real_out = tf.nn.relu(real_out)
            imag_out = tf.nn.relu(imag_out)
            tf_output = tf.complex(real_out, imag_out)
        elif self.activation == 'zrelu':
            tf_output = zrelu(tf.complex(real_out, imag_out))
        elif self.activation == 'modrelu':
            tf_output = modrelu(tf.complex(real_out, imag_out))
        elif self.activation == 'cardioid':
            tf_output = cardioid(tf.complex(real_out, imag_out))
        elif self.activation == 'last_layer':
            real_out = tf.nn.tanh(real_out)
            imag_out = tf.nn.relu(imag_out)
            tf_output = tf.complex(real_out, imag_out)
        else:
            tf_output = tf.complex(real_out, imag_out)

        return tf_output

class complex_Conv2DTranspose(Layer):

    def __init__(self, filters, kernel_size, stride=1, padding='valid', output_padding=None, data_format=None,
                dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None, **kwargs):
        super(complex_Conv2DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = stride
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        kernel_shape = (1, 1) + (self.filters, input_shape[-1])

        # Create a trainable weight variable for this layer:
        self.kernel_xR= self.add_weight(name='kernel_xR',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        constraint=self.kernel_constraint)
        self.kernel_xI= self.add_weight(name='kernel_xI',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        trainable=True,
                                        constraint=self.kernel_constraint)

        super(complex_Conv2DTranspose, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4)
        self.built = True

    def call(self, x):
        def split_complex(x):
            return (tf.math.real(x), tf.math.imag(x))

        def _deconv_length(
            input_length,
            filter_size,
            padding,
            output_padding=None,
            stride=0,
            dilation=1,
            ):
            # Source: https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py
            assert padding in {"same", "valid", "full"}
            if input_length is None:
                return None

            # Get the dilated kernel size
            filter_size = filter_size + (filter_size - 1) * (dilation - 1)

            # Infer length if output padding is None, else compute the exact length
            if output_padding is None:
                if padding == "valid":
                    length = input_length * stride + max(filter_size - stride, 0)
                elif padding == "full":
                    length = input_length * stride - (stride + filter_size - 2)
                elif padding == "same":
                    length = input_length * stride

            else:
                if padding == "same":
                    pad = filter_size // 2
                elif padding == "valid":
                    pad = 0
                elif padding == "full":
                    pad = filter_size - 1

                length = (
                    (input_length - 1) * stride + filter_size - 2 * pad + output_padding
                )
            return length

        input_shape = K.shape(x)
        batch_size = input_shape[0]
        
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = x.shape[h_axis], x.shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        dil_rt_h, dil_rt_w = self.dilation_rate

        # Infer the dynamic output shape:
        # (https://github.com/davidtvs/Keras-LinkNet/blob/master/models/conv2d_transpose.py)
        out_height = _deconv_length(height, kernel_h, self.padding, self.output_padding, stride_h, dil_rt_h)
        out_width = _deconv_length(width, kernel_w, self.padding, self.output_padding, stride_w, dil_rt_w)
        if self.data_format == 'channels_first':
            output_shape = (
                batch_size, self.filters, out_height, out_width
            )
        else:
            output_shape = (
                batch_size, out_height, out_width, self.filters
            )

        x_real, x_imag = split_complex(x)

        tf_real_real=K.conv2d_transpose(x_real, output_shape=output_shape,
                                        kernel=self.kernel_xR, strides=self.strides, padding=self.padding,
                                        data_format=self.data_format, dilation_rate=self.dilation_rate)
        tf_imag_real=K.conv2d_transpose(x_imag, output_shape=output_shape,
                                        kernel=self.kernel_xR, strides=self.strides, padding=self.padding,
                                        data_format=self.data_format,
                                        dilation_rate=self.dilation_rate)
        tf_real_imag=K.conv2d_transpose(x_real, output_shape=output_shape,
                                        kernel=self.kernel_xI, strides=self.strides, padding=self.padding,
                                        data_format=self.data_format,
                                        dilation_rate=self.dilation_rate)
        tf_imag_imag=K.conv2d_transpose(x_imag, output_shape=output_shape,
                                        kernel=self.kernel_xI, strides=self.strides, padding=self.padding,
                                        data_format=self.data_format,
                                        dilation_rate=self.dilation_rate)

        real_out = tf_real_real - tf_imag_imag
        imag_out = tf_imag_real + tf_real_imag
        
        if self.activation == 'cReLU':
            real_out = tf.nn.relu(real_out)
            imag_out = tf.nn.relu(imag_out)
        
        tf_output = tf.complex(real_out, imag_out)

        return tf_output


class complex_MaxPool2D(Layer):

    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super(complex_MaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        if strides is None:
            self.strides = pool_size
        else:
            self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        super(complex_MaxPool2D, self).build(input_shape)

    def call(self, x):
        def _mag_phase(x):
            return (tf.math.abs(x), tf.math.angle(x))

        def _unravel_argmax(argmax, shape):
            output_list = []
            output_list.append(argmax // (shape[2] * shape[3]))
            output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
            return tf.stack(output_list)

        x_shape = x.get_shape()
        channels = x_shape[-1]
        bs = tf.shape(x)[0]

        x_mag, x_phase = _mag_phase(x)

        # Pool magnitude, and use the same indices to pool phase
        # (Source: https://stackoverflow.com/questions/48215969/usage-of-argmax-from-tf-nn-max-pool-with-argmax-tensorflow)
        y_mag, argmax =  tf.nn.max_pool_with_argmax(input=x_mag, ksize=self.pool_size, strides=self.strides,
                                                    padding=self.padding, include_batch_in_index=True)
        argmax = tf.cast(argmax,tf.int32)
        unraveld = _unravel_argmax(argmax, x_shape)
        indices = tf.transpose(unraveld,(1,2,3,4,0))
        t1 = tf.range(channels,dtype=argmax.dtype)[None, None, None, :, None]
        t2 = tf.tile(t1,multiples=(bs,) + tuple(indices.get_shape()[1:-2]) + (1,1))
        t3 = tf.concat((indices,t2),axis=-1)
        t4 = tf.range(tf.cast(bs, dtype=argmax.dtype))
        t5 = tf.tile(t4[:,None,None,None,None],(1,) + tuple(indices.get_shape()[1:-2].as_list()) + (channels,1))
        t6 = tf.concat((t5, t3), -1)
        y_phase = tf.cast(tf.gather_nd(x_phase,t6),tf.complex64)

        y = tf.cast(y_mag,tf.complex64) * tf.math.exp(1j*y_phase)
        return y


# class mod_cardioid(Layer):
    
#     def __init__(self,beta_initializer='zeros', beta_regularizer=None, beta_constraint=None, **kwargs):
#         super(mod_cardioid, self).__init__(**kwargs)
#         self.beta_initializer = tf.keras.initializers.get(beta_initializer)
#         self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
#         self.beta_constraing = tf.keras.constraints.get(beta_constraint)

#     @tf_utils.shape_type_conversion
#     def build(self, input_shape):
#         param_shape = list(input_shape[1:])
#         if self.shared_axes is not None:
#             for i in self.shared_axes:
#                 param_shape[i - 1] = 1
#         self.beta =self.add_weight(shape=param_shape,
#                                     name='alpha',
#                                     initializer=self.alpha_initializer,
#                                     regularizer=self.alpha_regularizer,
#                                     constraint=self.alpha_constraint)
#         # Set input spec
#         axes = {}
#         if self.shared_axes:
#             for i in range(1, len(input_shape)):
#                 if i not in self.shared_axes:
#                     axes[i] = input_shape[i]
#         self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
#         self.built = True

#     def call(self, inputs):
#         phase = tf.math.angle(x) + self.beta
#         scale = 0.5 * (1 + tf.math.cos(phase))
#         output = tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
#         return output


def zrelu(x):
    # x and tf_output are complex-valued
    phase = tf.math.angle(x)

    # Check whether phase <= pi/2
    le = tf.less_equal(phase, pi / 2)

    # if phase <= pi/2, keep it in comp
    # if phase > pi/2, throw it away and set comp equal to 0
    y = tf.zeros_like(x)
    x = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.greater_equal(phase, 0)

    # if phase >= 0, keep it
    # if phase < 0, throw it away and set output equal to 0
    output = tf.where(ge, x, y)

    return output


def zrelu_v2(x):
    # x and tf_output are complex-valued
    phase = tf.math.angle(x)

    # Check whether phase <= pi
    le = tf.less_equal(phase, pi)

    # if phase <= pi, keep it in comp
    # if phase > pi, throw it away and set comp equal to 0
    y = tf.zeros_like(x)
    x = tf.where(le, x, y)

    # Check whether phase >= 0
    ge = tf.greater_equal(phase, 0)

    # if phase >= 0, keep it
    # if phase < 0, throw it away and set output equal to 0
    output = tf.where(ge, x, y)

    return output


def modrelu(x, data_format="channels_last"):
    input_shape = tf.shape(x)
    if data_format == "channels_last":
        axis_z = 1
        axis_y = 2
        axis_c = 3
    else:
        axis_c = 1
        axis_z = 2
        axis_y = 3

    # Channel size
    shape_c = x.shape[axis_c]

    with tf.name_scope("bias") as scope:
        if data_format == "channels_last":
            bias_shape = (1, 1, 1, shape_c)
        else:
            bias_shape = (1, shape_c, 1, 1)
        bias = tf.get_variable(name=scope,
                               shape=bias_shape,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    # relu(|z|+b) * (z / |z|)
    norm = tf.abs(x)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    output = tf.complex(tf.math.real(x) * scale,
                        tf.math.imag(x) * scale)

    return output


def cardioid(x):
    phase = tf.math.angle(x)
    scale = (1/10) * 0.5 * (1 + tf.math.cos(phase))
    output = tf.complex(tf.math.real(x) * scale, tf.math.imag(x) * scale)
    # output = 0.5*(1+tf.cos(phase))*z

    return output
