import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

import DLlib as dl

def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization


def MEBCRN(input_shape=(6, 192, 192, 2),
           n_outputs=4,
           output_activation='tanh',
		   n_mebc_blocks=4,
		   n_res_blocks=9,
           n_downsamplings=0,
		   filters=64,
           MLFF=True,
           dropout=0.0,
           self_attention=False,
		   norm='instance_norm'):
    Norm = _get_norm_layer(norm)
    n_ech = input_shape[-4]

    if n_res_blocks <= 1:
    	raise(ValueError("There should be at least 2 residual blocks"))

    def _residual_block(x):
        dim = x.shape[-1]
        h = x
        h = keras.layers.Conv2D(dim,3,padding='same',kernel_initializer='he_normal',activation='relu')(h)
        h = keras.layers.Conv2D(dim,3,padding='same',kernel_initializer='he_normal',activation='relu')(h)
        h = Norm()(h)
        h = keras.layers.add([x, h])
        return h

    def _MEBC_block(x,n_ech,nf,F_prev=None):
        x_vec = []
        F_prev_vec = []
        for ech in range(n_ech):
            nf1,nf2 = nf*ech,nf*(ech+1)
            # x splitting
            x_ech = keras.layers.Lambda(lambda x: x[:,ech:ech+1,:,:,:])(x)
            x_vec.append(x_ech)
            # F_prev splitting
            if F_prev is not None:
                F_prev_ech = keras.layers.Lambda(lambda x: x[:,ech,:,:,:])(F_prev)
                F_prev_vec.append(F_prev_ech)

        # Forward
        def _mid_echo_frw(xi,F_frw_im,F_prev_i=None):
            Fi_frw_A = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(xi)
            Fi_frw_B = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(F_frw_im)
            if F_prev_i is not None:
                Fi_frw_C = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(F_prev_i)
                Fi_frw = keras.layers.add([Fi_frw_A,Fi_frw_B,Fi_frw_C])
            else:
                Fi_frw = keras.layers.add([Fi_frw_A,Fi_frw_B])
            Fi_frw = tf.nn.relu(Fi_frw)
            Fi_frw = Norm()(Fi_frw)
            return Fi_frw

        x1 = x_vec[0] # (nb,hgt,wdt,nch)
        if F_prev is not None:
            F_prev_1 = F_prev_vec[0]
            F1_frw_A = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(x1)
            F1_frw_C = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(F_prev_1)
            F1_frw = keras.layers.add([F1_frw_A,F1_frw_C])
        else:
            F1_frw = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(x1)
        F1_frw = tf.nn.relu(F1_frw)
        F1_frw = Norm()(F1_frw)

        F_frw = [F1_frw]
        for k in range(n_ech-1):
            x_ech = x_vec[k+1]
            if F_prev is not None:
                F_prev_ech = F_prev_vec[k+1]
                F_frw_add = _mid_echo_frw(x_ech,F_frw[k],F_prev_ech)
            else:
                F_frw_add = _mid_echo_frw(x_ech,F_frw[k])
            F_frw.append(F_frw_add)

        # Reverse
        def _mid_echo_rev(xi,F_rev_ip,F_prev_i=None):
            Fi_rev_A = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(xi)
            Fi_rev_B = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(F_rev_ip)
            if F_prev_i is not None:
                Fi_rev_C = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(F_prev_i)
                Fi_rev = keras.layers.add([Fi_rev_A,Fi_rev_B,Fi_rev_C])
            else:
                Fi_rev = keras.layers.add([Fi_rev_A,Fi_rev_B])
            Fi_rev = tf.nn.relu(Fi_rev)
            Fi_rev = Norm()(Fi_rev)
            return Fi_rev

        x6 = x_vec[-1]
        if F_prev is not None:
            F_prev_6 = F_prev_vec[-1]
            F6_rev_A = keras.layers.Conv2D(nf,3,padding='same',use_bias=True)(x6)
            F6_rev_C = keras.layers.Conv2D(nf,3,padding='same',use_bias=True)(F_prev_6)
            F6_rev = keras.layers.add([F6_rev_A,F6_rev_C])
        else:
            F6_rev = keras.layers.Conv2D(nf,3,padding='same',kernel_initializer='he_normal')(x6)
        F6_rev = tf.nn.relu(F6_rev)
        F6_rev = Norm()(F6_rev)

        F_rev = [F6_rev]
        for k in range(n_ech-1):
            x_ech = x_vec[n_ech-2-k]
            if F_prev is not None:
                F_prev_ech = F_prev_vec[n_ech-2-k]
                F_rev_add = _mid_echo_rev(x_ech,F_rev[k],F_prev_ech)
            else:
                F_rev_add = _mid_echo_rev(x_ech,F_rev[k])
            F_rev.append(F_rev_add)
        F_rev.reverse()

        # Concatenate
        F = []
        for k in range(n_ech):
            x_ech = x_vec[k] # nb,hgt,wdt,2
            F_frw_ech = F_frw[k]
            F_rev_ech = F_rev[k]
            F_add_k = keras.layers.add([F_frw_ech,F_rev_ech])
            F_k = keras.layers.concatenate([F_add_k,x_ech]) # nb,1,hgt,wdt,(nf+2)
            F.append(F_k)

        return keras.layers.concatenate(F,axis=-4)

    def _MLFF_block(F_list):
        F_all = []
        for q in range(len(F_list)-1):
            F_aux_q = keras.layers.Conv2D(F_list[q].shape[-1],3,padding='same',kernel_initializer='he_normal',activation='relu')(F_list[q])
            F_all.append(F_aux_q)
        F_all.append(F_list[-1])
        return keras.layers.add(F_all)

    # 0
    h = inputs = keras.Input(shape=input_shape) # nb,ne,hgt,wdt,2

    # MEBCN
    nfb = filters
    h = _MEBC_block(h,n_ech,nfb)
    for i in range(n_mebc_blocks-1):
        nfb = filters//8 + (2*i)
        h = _MEBC_block(inputs,n_ech,nfb,h)

    h = keras.layers.Permute((2,3,1,4))(h) 
    h = keras.layers.Reshape((h.shape[-4],h.shape[-3],h.shape[-2]*h.shape[-1]))(h)

    dim = h.shape[-1]
    nfb = filters
    for _ in range(n_downsamplings):
        dim *= 2
        nfb *= 2
        h = keras.layers.Conv2D(dim,3,strides=2,padding='same')(h)
        h = tf.nn.relu(h)

    RB_list = []
    for _ in range(n_res_blocks):
        h = _residual_block(h)
        RB_list.append(h)

    if MLFF:
        h = _MLFF_block(RB_list)

    if self_attention:
        h = dl.SelfAttention(ch=h.shape[-1])(h)

    for _ in range(n_downsamplings):
        dim //= 2
        nfb //= 2
        h = keras.layers.Conv2DTranspose(dim,3,strides=2,padding='same',use_bias=False)(h)
        h = tf.nn.relu(h)

    h = keras.layers.Conv2D(filters,3,padding='same',kernel_initializer='he_normal')(h)
    output = keras.layers.Conv2D(n_outputs,3,padding='same',activation=output_activation,kernel_initializer='glorot_normal')(h)

    return keras.Model(inputs=inputs, outputs=output)