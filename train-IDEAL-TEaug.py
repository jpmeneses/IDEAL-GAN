import functools

import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import DLlib as dl
import pylib as py

import wflib as wf
import data

from itertools import cycle

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='WF-IDEAL')
py.arg('--DL_gen', type=bool, default=False)
py.arg('--DL_aug', type=bool, default=False)
py.arg('--DL_experiment_dir', default='output/GAN-238')
py.arg('--n_per_epoch', type=int, default=10000)
py.arg('--n_echoes', type=int, default=6)
py.arg('--field', type=float, default=1.5)
py.arg('--G_model', default='multi-decod', choices=['multi-decod','U-Net','MEBCRN'])
py.arg('--out_vars', default='WF', choices=['WF','WFc','PM','WF-PM'])
py.arg('--te_input', type=bool, default=True)
py.arg('--n_G_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--FM_aug', type=bool, default=False)
py.arg('--FM_mean', type=float, default=1.0)
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--FM_TV_weight', type=float, default=0.0)
py.arg('--R2_L1_weight', type=float, default=0.0)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=False)
py.arg('--D2_SelfAttention',type=bool, default=True)
py.arg('--D3_SelfAttention',type=bool, default=True)
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

if args.DL_gen or args.DL_aug:
    DL_args = py.args_from_yaml(py.join(args.DL_experiment_dir, 'settings.yml'))


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx, 
                            acqs_data=True, te_data=False, MEBCRN=True)

if not(args.DL_gen):
    dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
    acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True)

    dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
    acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True)

    dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
    acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True)

    dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
    acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# n1_div = 248
# n3_div = 0
# n4_div = 434

if args.DL_gen:
    trainX = trainY = np.zeros((args.n_per_epoch,1,1,1,1),dtype=np.float32)
else:
    trainX = np.concatenate((acqs_2,acqs_3,acqs_4,acqs_5),axis=0)
    trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4,out_maps_5),axis=0)
valX    = acqs_1
valY    = out_maps_1

# Overall dataset statistics
len_dataset,_,_,_,_ = np.shape(trainY)
_,n_out,hgt,wdt,n_ch = np.shape(valY)
echoes = args.n_echoes

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)

# Input and output dimensions (validations data)
print('Output shape:',valY.shape)

A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
A_B_dataset = A_B_dataset.batch(args.batch_size).shuffle(len(trainY))
A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

if args.G_model == 'multi-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(echoes,hgt,wdt,n_ch),
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                filters=args.n_G_filters,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(echoes,hgt,wdt,n_ch),
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)
elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        nn_out = 4
    else:
        nn_out = 2
    G_A2B = dl.UNet(input_shape=(echoes,hgt,wdt,n_ch),
                    n_out=nn_out,
                    te_input=args.te_input,
                    te_shape=(args.n_echoes,),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
elif args.G_model == 'MEBCRN':
    if args.out_vars=='WFc':
        nn_out = 4
        out_activ = None
    else:
        nn_out = 2
        out_activ = 'sigmoid'
    G_A2B=dl.MEBCRN(input_shape=(echoes,hgt,wdt,n_ch),
                    n_outputs=nn_out,
                    output_activation=out_activ,
                    n_res_blocks=9,
                    n_downsamplings=0,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
else:
    raise(NameError('Unrecognized Generator Architecture'))

if args.DL_gen or args.DL_aug:
    enc= dl.encoder(input_shape=(DL_args.n_echoes,hgt,wdt,n_ch),
                    encoded_dims=DL_args.encoded_size,
                    filters=DL_args.n_G_filters,
                    num_layers=DL_args.n_downsamplings,
                    num_res_blocks=DL_args.n_res_blocks,
                    kl_reg=False,
                    NL_self_attention=DL_args.NL_SelfAttention
                    )
    dec_w =  dl.decoder(encoded_dims=DL_args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=DL_args.n_G_filters,
                        num_layers=DL_args.n_downsamplings,
                        num_res_blocks=DL_args.n_res_blocks,
                        NL_self_attention=DL_args.NL_SelfAttention
                        )
    dec_f =  dl.decoder(encoded_dims=DL_args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=DL_args.n_G_filters,
                        num_layers=DL_args.n_downsamplings,
                        num_res_blocks=DL_args.n_res_blocks,
                        NL_self_attention=DL_args.NL_SelfAttention
                        )
    dec_xi = dl.decoder(encoded_dims=DL_args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        n_groups=DL_args.n_groups_PM,
                        filters=DL_args.n_G_filters,
                        num_layers=DL_args.n_downsamplings,
                        num_res_blocks=DL_args.n_res_blocks,
                        NL_self_attention=DL_args.NL_SelfAttention
                        )
    tl.Checkpoint(dict(enc=enc,dec_w=dec_w,dec_f=dec_f,dec_xi=dec_xi), py.join(args.DL_experiment_dir, 'checkpoints')).restore()

IDEAL_op = wf.IDEAL_Layer(field=args.field)

sup_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B, te=None):
    with tf.GradientTape() as t:
        ##################### B Cycle #####################
        if not(args.DL_aug):
            B2A = IDEAL_op(B, te=te, training=False)
            B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
        else:
            A2Z = enc(A, training=False)
            A2Z = keras.layers.GaussianNoise(stddev=0.05)(A2Z)
            A2Z2B_w = dec_w(A2Z, training=False)
            A2Z2B_f = dec_f(A2Z, training=False)
            A2Z2B_xi= dec_xi(A2Z, training=False)
            B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)
            B2A = IDEAL_op(B, te=te, training=False)

        B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
        B_PM = B[:,2:,:,:,:]

        if args.out_vars == 'WF':
            # Compute model's output
            if args.te_input:
                B2A2B_WF_abs = G_A2B([B2A,te], training=True)
            else:
                B2A2B_WF_abs = G_A2B(B2A, training=True)
            B2A2B_WF_abs = tf.where(B[:,:2,:,:,:1]!=0.0,B2A2B_WF_abs,0.0)

            # Compute zero-valued param maps
            B2A2B_PM = tf.zeros_like(B_PM)

            # Split A2B param maps
            B2A2B_R2 = B2A2B_PM[:,:,:,:,1:]
            B2A2B_FM = B2A2B_PM[:,:,:,:,:1]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF_abs, B2A2B_WF_abs)

        elif args.out_vars == 'WFc':
            # Compute model's output
            if args.te_input:
                B2A2B_WF = G_A2B([B2A,te], training=True)
            else:
                B2A2B_WF = G_A2B(B2A, training=True)
            B2A2B_WF = tf.where(B[:,:2,:,:,:]!=0.0,B2A2B_WF,0.0)

            # Magnitude of water/fat images
            B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))

            # Compute zero-valued param maps
            B2A2B_PM = tf.zeros_like(B[:,2:,:,:,:])
            B2A2B_R2 = B2A2B_PM[:,:,:,:,1:]
            B2A2B_FM = B2A2B_PM[:,:,:,:,:1]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF, B2A2B_WF)

        elif args.out_vars == 'PM':
            # Compute model's output
            if args.te_input:
                B2A2B_PM = G_A2B([B2A,te], training=True)
            else:
                B2A2B_PM = G_A2B(B2A, training=True)
            if not(args.DL_gen):
                B2A2B_PM = tf.where(B_PM!=0.0,B2A2B_PM,0.0)

            # Split A2B param maps
            B2A2B_R2 = B2A2B_PM[:,0,:,:,1:]
            B2A2B_FM = B2A2B_PM[:,0,:,:,:1]

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                B2A2B_FM = (B2A2B_FM - 0.5) * 2
                if not(args.DL_gen):
                    B2A2B_FM = tf.where(B[:,:,:,:1]!=0.0,B2A2B_FM,0.0)
                B2A2B_PM = tf.concat([B2A2B_R2,B2A2B_FM],axis=-1)

            # Compute water/fat
            B2A2B_WF = wf.get_rho(B2A, B2A2B_PM, field=args.field, te=te)
            
            # Magnitude of water/fat images
            B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))
            
            # Compute loss
            sup_loss = sup_loss_fn(B_PM, B2A2B_PM)

        # elif args.out_vars == 'WF-PM':
        #     # Compute model's output
        #     B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
        #     if args.te_input:
        #         B2A2B_abs = G_A2B([B2A,te], training=True)
        #     else:
        #         B2A2B_abs = G_A2B(B2A, training=True)
        #     B2A2B_abs = tf.where(B[:,:,:,:4]!=0.0,B2A2B_abs,0.0)

        #     # Split A2B outputs
        #     B2A2B_WF_abs, B2A2B_PM = tf.dynamic_partition(B2A2B_abs,indx_B_abs,num_partitions=2)
        #     B2A2B_WF_abs = tf.reshape(B2A2B_WF_abs,B[:,:,:,:2].shape)
        #     B2A2B_PM = tf.reshape(B2A2B_PM,B[:,:,:,4:].shape)

        #     # Split A2B param maps
        #     B2A2B_R2, B2A2B_FM = tf.dynamic_partition(B2A2B_PM,indx_PM,num_partitions=2)
        #     B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
        #     B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)

        #     # Restore field-map when necessary
        #     if args.G_model=='U-Net' or args.G_model=='MEBCRN':
        #         B2A2B_FM = (B2A2B_FM - 0.5) * 2
        #         B2A2B_FM = tf.where(B[:,:,:,:1]!=0.0,B2A2B_FM,0.0)
        #         B2A2B_abs = tf.concat([B2A2B_WF_abs,B2A2B_R2,B2A2B_FM],axis=-1)

        #     # Compute loss
        #     sup_loss = sup_loss_fn(B_abs, B2A2B_abs)

        ############### Splited losses ####################
        WF_abs_loss = sup_loss_fn(B_WF_abs, B2A2B_WF_abs)
        R2_loss = sup_loss_fn(B[:,2,:,:,1:], B2A2B_R2)
        FM_loss = sup_loss_fn(B[:,2,:,:,:1], B2A2B_FM)

        ################ Regularizers #####################
        R2_TV = tf.reduce_sum(tf.image.total_variation(B2A2B_R2)) * args.R2_TV_weight
        FM_TV = tf.reduce_sum(tf.image.total_variation(B2A2B_FM)) * args.FM_TV_weight
        R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(B2A2B_R2),axis=(1,2,3))) * args.R2_L1_weight
        FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(B2A2B_FM),axis=(1,2,3))) * args.FM_L1_weight
        reg_term = R2_TV + FM_TV + R2_L1 + FM_L1
        
        G_loss = sup_loss + reg_term
        
    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return {'sup_loss': sup_loss,
            'WF_loss': WF_abs_loss,
            'R2_loss': R2_loss,
            'FM_loss': FM_loss,
            'TV_R2': R2_TV,
            'TV_FM': FM_TV,
            'L1_R2': R2_L1,
            'L1_FM': FM_L1}


def train_step(A, B, te=None):
    G_loss_dict = train_G(A, B, te)
    return G_loss_dict


@tf.function
def sample(A, B, te=None):
    if args.DL_aug:
        A2Z = enc(A, training=False)
        A2Z = keras.layers.GaussianNoise(stddev=0.05)(A2Z)
        A2Z2B_w = dec_w(A2Z, training=False)
        A2Z2B_f = dec_f(A2Z, training=False)
        A2Z2B_xi= dec_xi(A2Z, training=False)
        B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)

    # Split B
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
    B_PM = B[:,2:,:,:,:]

    # Compute B2A (+ noise) and estimate B2A2B
    B2A = IDEAL_op(B, te=te, training=False)
    B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
    
    # Estimate A2B
    if args.out_vars == 'WF':
        if args.te_input:
            B2A2B_WF_abs = G_A2B([B2A,te], training=True)
        else:
            B2A2B_WF_abs = G_A2B(B2A, training=True)
        B2A2B_WF_abs = tf.where(B_WF_abs!=0.0,B2A2B_WF_abs,0.0)
        B2A2B_WF = tf.concat([B2A2B_WF_abs, tf.zeros_like(B2A2B_WF_abs)], axis=1)
        B2A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        B2A2B_R2 = B2A2B_PM[:,:,:,:,1:]
        B2A2B_FM = B2A2B_PM[:,:,:,:,:1]
        B2A2B_abs = tf.concat([B2A2B_WF,B2A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF_abs, B2A2B_WF_abs)
    # elif args.out_vars == 'WFc':
    #     if args.te_input:
    #         B2A2B_WF = G_A2B([B2A,te], training=True)
    #     else:
    #         B2A2B_WF = G_A2B(B2A, training=True)
    #     B2A2B_WF = tf.where(B[:,:,:,:4]!=0.0,B2A2B_WF,0.0)
    #     B2A2B_WF_real = B2A2B_WF[:,:,:,0::2]
    #     B2A2B_WF_imag = B2A2B_WF[:,:,:,1::2]
    #     B2A2B_WF_abs = tf.abs(tf.complex(B2A2B_WF_real,B2A2B_WF_imag))
    #     B2A2B_PM = tf.zeros_like(B_PM)
    #     # Split A2B param maps
    #     B2A2B_R2, B2A2B_FM = tf.dynamic_partition(B2A2B_PM,indx_PM,num_partitions=2)
    #     B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
    #     B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
    #     B2A2B_abs = tf.concat([B2A2B_WF_abs,B2A2B_PM],axis=-1)
    #     val_sup_loss = sup_loss_fn(B_WF, B2A2B_WF)
    elif args.out_vars == 'PM':
        if args.te_input:
            B2A2B_PM = G_A2B([B2A,te], training=True)
        else:
            B2A2B_PM = G_A2B(B2A, training=True)
        B2A2B_PM = tf.where(B_PM!=0.0,B2A2B_PM,0.0)
        B2A2B_R2 = B2A2B_PM[:,:,:,:,1:]
        B2A2B_FM = B2A2B_PM[:,:,:,:,:1]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            B2A2B_FM = (B2A2B_FM - 0.5) * 2
            B2A2B_FM = tf.where(B_PM[:,:,:,:,:1]!=0.0,B2A2B_FM,0.0)
            B2A2B_PM = tf.concat([B2A2B_R2,B2A2B_FM],axis=1)
        B2A2B_WF = wf.get_rho(B2A, B2A2B_PM, field=args.field, te=te)
        B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))
        B2A2B = tf.concat([B2A2B_WF,B2A2B_PM],axis=1)
        val_sup_loss = sup_loss_fn(B_PM, B2A2B_PM)
    # elif args.out_vars == 'WF-PM':
    #     B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
    #     if args.te_input:
    #         B2A2B_abs = G_A2B([B2A,te], training=True)
    #     else:
    #         B2A2B_abs = G_A2B(B2A, training=True)
    #     B2A2B_abs = tf.where(B_abs!=0.0,B2A2B_abs,0.0)
    #     B2A2B_WF_abs,B2A2B_PM = tf.dynamic_partition(B2A2B_abs,indx_B_abs,num_partitions=2)
    #     B2A2B_WF_abs = tf.reshape(B2A2B_WF_abs,B[:,:,:,:2].shape)
    #     B2A2B_PM = tf.reshape(B2A2B_PM,B[:,:,:,4:].shape)
    #     B2A2B_R2, B2A2B_FM = tf.dynamic_partition(B2A2B_PM,indx_PM,num_partitions=2)
    #     B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
    #     B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
    #     if args.G_model=='U-Net' or args.G_model=='MEBCRN':
    #         B2A2B_FM = (B2A2B_FM - 0.5) * 2
    #         B2A2B_FM = tf.where(B_PM[:,:,:,:1]!=0.0,B2A2B_FM,0.0)
    #         B2A2B_abs = tf.concat([B2A2B_WF_abs,B2A2B_R2,B2A2B_FM],axis=-1)
    #     val_sup_loss = sup_loss_fn(B_abs, B2A2B_abs)

    ############### Splited losses ####################
    WF_abs_loss = sup_loss_fn(B_WF_abs, B2A2B_WF_abs)
    R2_loss = sup_loss_fn(B[:,2:,:,:,1:], B2A2B_R2)
    FM_loss = sup_loss_fn(B[:,2:,:,:,:1], B2A2B_FM)
    
    return B2A, B2A2B, {'sup_loss': val_sup_loss,
                        'WF_loss': WF_abs_loss,
                        'R2_loss': R2_loss,
                        'FM_loss': FM_loss}

def validation_step(A, B, TE):
    B2A, B2A2B_abs, val_B2A2B_dict = sample(A, B, TE)
    return B2A, B2A2B_abs, val_B2A2B_dict

if args.DL_gen:
    @tf.function
    def gen_sample(Z,TE=None):
        # Z2B2A Cycle
        Z2B_w = dec_w(Z, training=False)
        Z2B_f = dec_f(Z, training=False)
        Z2B_xi= dec_xi(Z, training=False)
        Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
        return Z2B

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_optimizer=G_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))
val_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'validation'))

# sample
val_iter = cycle(A_B_dataset_val)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)
n_div = np.ceil(total_steps/len(valY))

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for A, B in A_B_dataset:
        if args.DL_gen:
            hls = hgt//(2**(DL_args.n_downsamplings))
            wls = wdt//(2**(DL_args.n_downsamplings))
            z_shape = (1,hls,wls,DL_args.encoded_size)
            Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
            B = gen_sample(Z)
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        p = np.random.rand()
        if p <= 0.4:
            A = tf.reshape(tf.transpose(A,perm=[0,2,3,1,4]),[A.shape[0],hgt,wdt,args.n_echoes*n_ch])
            B = tf.reshape(tf.transpose(B,perm=[0,2,3,1,4]),[B.shape[0],hgt,wdt,n_out*n_ch])

            # Random 90 deg rotations
            A = tf.image.rot90(A,k=np.random.randint(3))
            B = tf.image.rot90(B,k=np.random.randint(3))

            # Random horizontal reflections
            A = tf.image.random_flip_left_right(A)
            B = tf.image.random_flip_left_right(B)

            # Random vertical reflections
            A = tf.image.random_flip_up_down(A)
            B = tf.image.random_flip_up_down(B)

            A = tf.transpose(tf.reshape(A,[A.shape[0],hgt,wdt,args.n_echoes,n_ch]),[0,3,1,2,4])
            B = tf.transpose(tf.reshape(B,[B.shape[0],hgt,wdt,n_out,n_ch]),[0,3,1,2,4])

            # Random off-resonance field-map scaling factor
            if args.FM_aug:
                B_FM = B[:,2:,:,:,:1] * tf.random.normal([1],mean=args.FM_mean,stddev=0.25,dtype=tf.float32)
                B_PM = tf.concat([B_FM,B[:,2:,:,:,1:]], axis=-1)
                B = tf.concat([B[:,:2,:,:,:],B_PM], axis=1)
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        if args.field == 3.0:
            te_var = wf.gen_TEvar(args.n_echoes, bs=B.shape[0], TE_ini_d=0.4e-3, d_TE_min=1.0e-3, d_TE_d=0.3e-3)
        else:
            te_var = wf.gen_TEvar(args.n_echoes, bs=B.shape[0])

        G_loss_dict = train_step(A, B, te=te_var)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200):
            A, B = next(val_iter)
            A = tf.expand_dims(A, axis=0)
            B = tf.expand_dims(B, axis=0)
            B_WF = B[:,:2,:,:,:]
            B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))

            if args.field == 3.0:
                TE_valid = wf.gen_TEvar(args.n_echoes, 1, TE_ini_d=0.4e-3, d_TE_min=1.0e-3, d_TE_d=0.3e-3)
            else:
                TE_valid = wf.gen_TEvar(args.n_echoes, 1)
            
            B2A, B2A2B, val_A2B_dict = validation_step(A, B, TE_valid)
            B2A2B_WF = B2A2B[:,:2,:,:,:]
            B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_A2B_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(B2A[:,0,:,:,0],B2A[:,0,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(B2A[:,1,:,:,0],B2A[:,1,:,:,1])))
            if args.n_echoes >= 3:
                im_ech3 = np.squeeze(np.abs(tf.complex(B2A[:,2,:,:,0],B2A[:,2,:,:,1])))
            if args.n_echoes >= 4:
                im_ech4 = np.squeeze(np.abs(tf.complex(B2A[:,3,:,:,0],B2A[:,3,:,:,1])))
            if args.n_echoes >= 5:
                im_ech5 = np.squeeze(np.abs(tf.complex(B2A[:,4,:,:,0],B2A[:,4,:,:,1])))
            if args.n_echoes >= 6:
                im_ech6 = np.squeeze(np.abs(tf.complex(B2A[:,5,:,:,0],B2A[:,5,:,:,1])))
            
            # Acquisitions in the first row
            acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,0].set_title('1st Echo')
            axs[0,0].axis('off')
            acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,1].set_title('2nd Echo')
            axs[0,1].axis('off')
            if args.n_echoes >= 3:
                acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,2].set_title('3rd Echo')
                axs[0,2].axis('off')
            else:
                fig.delaxes(axs[0,2])
            if args.n_echoes >= 4:
                acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,3].set_title('4th Echo')
                axs[0,3].axis('off')
            else:
                fig.delaxes(axs[0,3])
            if args.n_echoes >= 5:
                acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,4].set_title('5th Echo')
                axs[0,4].axis('off')
            else:
                fig.delaxes(axs[0,4])
            if args.n_echoes >= 6:
                acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,5].set_title('6th Echo')
                axs[0,5].axis('off')
            else:
                fig.delaxes(axs[0,5])

            # B2A2B maps in the second row
            w_aux = np.squeeze(B2A2B_WF_abs[:,0,:,:,:])
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(B2A2B_WF_abs[:,1,:,:,:])
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(B2A2B[:,2,:,:,1])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(B2A2B[:,2,:,:,0])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,4])
            axs[1,4].axis('off')
            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            wn_aux = np.squeeze(B_WF_abs[:,0,:,:,:])
            W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            fn_aux = np.squeeze(B2A2B_WF_abs[:,1,:,:,:])
            F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            r2n_aux = np.squeeze(B[:,2,:,:,1])
            r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            fieldn_aux = np.squeeze(B[:,2,:,:,0])
            field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_unet, ax=axs[2,4])
            axs[2,4].axis('off')
            fig.delaxes(axs[2,0])
            fig.delaxes(axs[2,5])

            fig.suptitle('TE1/dTE: '+str([TE_valid[0,0,0].numpy(),np.mean(np.diff(TE_valid,axis=1))]), fontsize=16)

            # plt.show()
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
