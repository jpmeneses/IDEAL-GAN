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
py.arg('--data_size', type=int, default=192, choices=[192,384])
py.arg('--DL_gen', type=bool, default=False)
py.arg('--DL_partial_real', type=int, default=0, choices=[0,2,6,10])
py.arg('--DL_filename', default='LDM_ds')
py.arg('--sigma_noise', type=float, default=0.0)
py.arg('--shuffle', type=bool, default=True)
py.arg('--n_echoes', type=int, default=6)
py.arg('--bip_grad', type=bool, default=False)
py.arg('--field', type=float, default=1.5)
py.arg('--G_model', default='multi-decod', choices=['multi-decod','U-Net','2U-Net'])
py.arg('--out_vars', default='WF', choices=['WF','WFc','PM','WF-PM'])
py.arg('--te_input', type=bool, default=True)
py.arg('--n_G_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=20)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--sel_weight', type=bool, default=False)
py.arg('--sel_weight_pwr', type=float, default=1.0)
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

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

if args.n_echoes > 0:
    ech_idx = args.n_echoes * 2
    echoes = args.n_echoes
else:
    ech_idx = 12
    echoes = None
if args.bip_grad:
    bip_out = 1
else:
    bip_out = 0
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'

dataset_hdf5_1 = 'INTA_GC_' + str(args.data_size) + '_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx,
                            acqs_data=True, te_data=False, MEBCRN=True)

A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

if not(args.DL_gen):
    dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                acqs_data=False, te_data=False, MEBCRN=False)

    dataset_hdf5_3 = 'Volunteers_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    out_maps_3 = data.load_hdf5(dataset_dir, dataset_hdf5_3, ech_idx,
                                acqs_data=False, te_data=False, MEBCRN=False)

    dataset_hdf5_4 = 'Attilio_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    out_maps_4 = data.load_hdf5(dataset_dir, dataset_hdf5_4, ech_idx,
                                acqs_data=False, te_data=False, MEBCRN=False)

    trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4),axis=0)

    if args.G_model == 'MEBCRN':
        len_dataset,n_out,hgt,wdt,n_ch = np.shape(valY)
    else:
        len_dataset,hgt,wdt,n_out = np.shape(valY)
        n_ch = 2

    B_dataset = tf.data.Dataset.from_tensor_slices(trainY)

else:
    recordPath = py.join('tfrecord', args.DL_filename)
    tfr_dataset = tf.data.TFRecordDataset([recordPath])
    # Create a description of the features.
    feature_description = {
        'acqs': tf.io.FixedLenFeature([], tf.string),
        'out_maps': tf.io.FixedLenFeature([], tf.string),
        }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        parsed_ds = tf.io.parse_example(example_proto, feature_description)
        return tf.io.parse_tensor(parsed_ds['out_maps'], out_type=tf.float32)

    if args.DL_partial_real != 0:
        if args.TE1 == 0.0014 and args.dTE == 0.0022:
            dataset_hdf5_1 = 'multiTE_' + str(args.data_size) + '_complex_2D.hdf5'
            ini_idxs = [0,84,204,300,396,484,580,680,776,848]#,932,1028, 1100,1142,1190,1232,1286,1334,1388,1460]
            delta_idxs = [21,24,24,24,22,24,25,24,18]#,21,24,18, 21,24,21,18,16,18,24,21]
            end_idx = np.sum(delta_idxs)
            k_idxs = [(0,1),(2,3)]
            for k in k_idxs:
                custom_list = [a for a in range(ini_idxs[0]+k[0]*delta_idxs[0],ini_idxs[0]+k[1]*delta_idxs[0])]
            # Rest of the patients
            for i in range(1,len(ini_idxs)):
                if (i<=11) and args.TE1 == 0.0013 and args.dTE == 0.0022:
                    k_idxs = [(0,1),(2,3)]
                elif (i<=11) and args.TE1 == 0.0014 and args.dTE == 0.0022:
                    k_idxs = [(0,1),(3,4)]
                elif (i==1) and args.TE1 == 0.0013 and args.dTE == 0.0023:
                    k_idxs = [(0,1),(4,5)]
                elif (i==15 or i==16) and args.TE1 == 0.0013 and args.dTE == 0.0023:
                    k_idxs = [(0,1),(2,3)]
                elif (i>=17) and args.TE1 == 0.0013 and args.dTE == 0.0024:
                    k_idxs = [(0,1),(2,3)]
                else:
                    k_idxs = [(0,2)]
                for k in k_idxs:
                    custom_list += [a for a in range(ini_idxs[i]+k[0]*delta_idxs[i],ini_idxs[i]+k[1]*delta_idxs[i])]
                trainX, trainY, TEs =data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, custom_list=custom_list,
                                                    acqs_data=True,te_data=True,remove_zeros=False,
                                                    MEBCRN=True, mag_and_phase=True, unwrap=True)
        else:
            if args.DL_partial_real == 2:
                end_idx = 62
            elif args.DL_partial_real == 6:
                end_idx = 200
            elif args.DL_partial_real == 10:
                end_idx = 330
            dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
            trainY = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx, end=end_idx,
                                    acqs_data=False, te_data=False, MEBCRN=False,
                                    mag_and_phase=True, unwrap=True)
        B_dataset = tfr_dataset.skip(end_idx).map(_parse_function)
        B_dataset_aux = tf.data.Dataset.from_tensor_slices(trainY)
        B_dataset = B_dataset.concatenate(B_dataset_aux)
    else:
        B_dataset = tfr_dataset.map(_parse_function)

    for B in B_dataset.take(1):
        _,hgt,wdt,n_ch = B.shape
    len_dataset = int(args.DL_filename.split('_')[-1])
    if args.DL_partial_real != 0:
        len_dataset += trainY.shape[0]

B_dataset = B_dataset.batch(args.batch_size)
if args.shuffle:
    B_dataset = B_dataset.shuffle(len_dataset)

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
                                te_shape=(echoes,),
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)
else:
    if args.out_vars == 'WF-PM':
        nn_out = 4
    elif args.G_model == '2U-Net':
        nn_out = 1
        G_A2R2= dl.UNet(input_shape=(echoes,hgt,wdt,1),
                        n_out=nn_out,
                        ME_layer=True,
                        te_input=args.te_input,
                        te_shape=(echoes,),
                        filters=args.n_G_filters,
                        output_activation='sigmoid',
                        self_attention=args.D2_SelfAttention)
    else:
        nn_out = 2
    G_A2B = dl.UNet(input_shape=(echoes,hgt,wdt,n_ch),
                    n_out=nn_out+bip_out,
                    ME_layer=True,
                    te_input=args.te_input,
                    te_shape=(echoes,),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
        

IDEAL_op = wf.IDEAL_Layer(field=args.field)

sup_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)
G_R2_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(B, te=None):
    B = data.B_to_MEBCRN(B)
    ##################### B Cycle #####################
    B2A = IDEAL_op(B, te=te, training=False)
    B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
    B2A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A),axis=-1,keepdims=True))

    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
    B_PM = B[:,2:,:,:,:]

    ############## Selective weighting ################
    if args.sel_weight:
        sel_w = 0.0
        for echo in range(3):
            obs_phase = tf.math.atan2(B2A[:,echo:(echo+1),:,:,1:],B2A[:,echo:(echo+1),:,:,:1])
            phi_phase = 2*np.pi*B[:,2:,:,:,:1]*fm_sc * te[0,echo,0]
            phi_phase += tf.math.atan2(B[:,:1,:,:,1:],B[:,:1,:,:,:1])
            sel_w += (1/6) * tf.math.cos(obs_phase-phi_phase) + (1/6)
            sel_w **= args.sel_weight_pwr
    else:
        sel_w = 1.0

    with tf.GradientTape() as t:
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
            if args.G_model == '2U-Net':
                if args.te_input:
                    B2A2B_R2 = G_A2R2([B2A_abs,te], training=False)
                else:
                    B2A2B_R2 = G_A2R2(B2A_abs, training=False)
                if args.bip_grad:
                    B2A2B_PM = tf.transpose(B2A2B_PM, perm=[0,4,2,3,1])
                    B2A2B_R2 = tf.concat([B2A2B_R2,tf.zeros_like(B2A2B_R2)],axis=1)
                B2A2B_PM = tf.concat([B2A2B_PM,B2A2B_R2],axis=-1)
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
            if args.sel_weight:
                sel_w = tf.concat([sel_w,sel_w],axis=-1)
            sup_loss = sup_loss_fn(sel_w*B_PM, sel_w*B2A2B_PM)

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

@tf.function
def train_G_R2(B, te=None):
    B = data.B_to_MEBCRN(B)
    ##################### B Cycle #####################
    B2A = IDEAL_op(B, te=te, training=False)
    B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
    B2A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A),axis=-1,keepdims=True))

    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
    B_PM = B[:,2:,:,:,:]

    with tf.GradientTape() as t:
        # Compute model's output
        if args.te_input:
            B2A2B_FM = G_A2B([B2A,te], training=False)
            B2A2B_R2 = G_A2R2([B2A_abs,te], training=True)
        else:
            B2A2B_FM = G_A2B(B2A, training=False)
            B2A2B_R2 = G_A2R2(B2A_abs, training=True)
        if args.bip_grad:
            B2A2B_FM = tf.transpose(B2A2B_FM, perm=[0,4,2,3,1])
            B2A2B_R2 = tf.concat([B2A2B_R2,tf.zeros_like(B2A2B_R2)],axis=1)
        B2A2B_PM = tf.concat([B2A2B_FM,B2A2B_R2],axis=-1)
        B2A2B_PM = tf.where(B_PM!=0.0,B2A2B_PM,0.0)

        # Compute water/fat
        B2A2B_WF = wf.get_rho(B2A, B2A2B_PM, field=args.field, te=te)
        
        # Magnitude of water/fat images
        B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))
        
        # Compute loss
        R2_loss = sup_loss_fn(B_PM[:,:1,:,:,1:], B2A2B_R2[:,:1,...])

        ############### Splited losses ####################
        WF_abs_loss = sup_loss_fn(B_WF_abs, B2A2B_WF_abs)

        ################ Regularizers #####################
        R2_TV = tf.reduce_sum(tf.image.total_variation(tf.squeeze(B2A2B_R2[:,:1,...],axis=1))) * args.R2_TV_weight
        R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(B2A2B_R2[:,:1,...]),axis=(1,2,3,4))) * args.R2_L1_weight
        reg_term = R2_TV + R2_L1
        
        G_loss = R2_loss + reg_term
        
    G_grad = t.gradient(G_loss, G_A2R2.trainable_variables)
    G_R2_optimizer.apply_gradients(zip(G_grad, G_A2R2.trainable_variables))

    return {'R2_loss': R2_loss,
            'TV_R2': R2_TV,
            'L1_R2': R2_L1}



def train_step(B, te=None):
    G_loss_dict = train_G(B, te)
    if args.G_model == '2U-Net':
        G_loss_dict_aux = train_G_R2(B, te)
        G_loss_dict.update(G_loss_dict_aux)
    return G_loss_dict


@tf.function
def sample(B, te=None):
    # Split B
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
    B_PM = B[:,2:,:,:,:]

    # Compute B2A (+ noise) and estimate B2A2B
    B2A = IDEAL_op(B, te=te, training=False)
    B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
    B2A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A),axis=-1,keepdims=True))
    
    # Estimate A2B
    if args.out_vars == 'WF':
        if args.te_input:
            B2A2B_WF_abs = G_A2B([B2A,te], training=False)
        else:
            B2A2B_WF_abs = G_A2B(B2A, training=False)
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
            B2A2B_PM = G_A2B([B2A,te], training=False)
        else:
            B2A2B_PM = G_A2B(B2A, training=False)
        if args.G_model == '2U-Net':
            if args.te_input:
                B2A2B_R2 = G_A2R2([B2A_abs,te], training=False)
            else:
                B2A2B_R2 = G_A2R2(B2A_abs, training=False)
            if args.bip_grad:
                B2A2B_PM = tf.transpose(B2A2B_PM, perm=[0,4,2,3,1])
                B2A2B_R2 = tf.concat([B2A2B_R2,tf.zeros_like(B2A2B_R2)],axis=1)
            B2A2B_PM = tf.concat([B2A2B_PM,B2A2B_R2],axis=-1)
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

def validation_step(B, TE):
    B2A, B2A2B_abs, val_B2A2B_dict = sample(B, TE)
    return B2A, B2A2B_abs, val_B2A2B_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
ckpt_dict = dict(G_A2B=G_A2B, G_optimizer=G_optimizer, ep_cnt=ep_cnt)
if args.G_model == '2U-Net':
    ckpt_dict_aux = dict(G_A2R2=G_A2R2, G_R2_optimizer=G_R2_optimizer)
    ckpt_dict.update(ckpt_dict_aux)
checkpoint = tl.Checkpoint(ckpt_dict,
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
    for B in B_dataset:
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        p = np.random.rand()
        if p <= 0.4:
            # Random 90 deg rotations
            B = tf.image.rot90(B,k=np.random.randint(3))

            # Random horizontal reflections
            B = tf.image.random_flip_left_right(B)

            # Random vertical reflections
            B = tf.image.random_flip_up_down(B)

            # Random off-resonance field-map scaling factor
            if args.FM_aug:
                B_FM = B[...,-1:] * tf.random.normal([1],mean=args.FM_mean,stddev=0.25,dtype=tf.float32)
                B = tf.concat([B[...,:-1],B_FM], axis=-1)

        if args.bip_grad:
            # TO BE DEBUGGED FOR non-MEBCRN FORMAT
            B_FM = B[:,2:,:,:,:1]
            x_lim = np.random.uniform(0.1,0.5)
            x_off = np.random.uniform(0.0,0.01)
            x = tf.linspace(-x_lim,x_lim,B_FM.shape[2]) + x_off
            X, Y = tf.meshgrid(x, x)
            B_bp = tf.ones_like(B_FM)
            B_bp = tf.where(B_FM!=0.0,tf.expand_dims(X,axis=-1),0.0)
            B_bp = tf.concat([B_bp,tf.zeros_like(B_bp)],axis=-1)
            B = tf.concat([B,B_bp],axis=1)
        
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        if args.n_echoes == 0:
            if args.bip_grad:
                ne_sel = np.random.randint(6,13)
            else:
                ne_sel = np.random.randint(2,7)
        else:
            ne_sel = 0
        if args.field == 3.0:
            te_var=wf.gen_TEvar(args.n_echoes+ne_sel, bs=B.shape[0], 
                                TE_ini_d=0.4e-3, d_TE_min=1.0e-3, d_TE_d=0.3e-3)
        elif args.bip_grad:
            te_var=wf.gen_TEvar(args.n_echoes+ne_sel, bs=B.shape[0],
                                d_TE_min=0.9e-3, d_TE_d=0.3e-3)
        else:
            te_var = wf.gen_TEvar(args.n_echoes+ne_sel, bs=B.shape[0])

        G_loss_dict = train_step(B, te=te_var)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200//args.batch_size):
            B = next(val_iter)
            B = tf.expand_dims(B, axis=0)
            B_WF = B[:,:2,:,:,:]
            B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))

            if args.field == 3.0:
                TE_valid = wf.gen_TEvar(args.n_echoes+ne_sel, 1, TE_ini_d=0.4e-3, d_TE_min=1.0e-3, d_TE_d=0.3e-3)
            elif args.bip_grad:
                TE_valid = wf.gen_TEvar(args.n_echoes+ne_sel, 1, d_TE_min=0.9e-3, d_TE_d=0.3e-3)
            else:
                TE_valid = wf.gen_TEvar(args.n_echoes+ne_sel, 1)
            
            B2A, B2A2B, val_A2B_dict = validation_step(B, TE_valid)
            B2A2B_WF = B2A2B[:,:2,:,:,:]
            B2A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B2A2B_WF),axis=-1,keepdims=True))

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_A2B_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(B2A[:,0,:,:,0],B2A[:,0,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(B2A[:,1,:,:,0],B2A[:,1,:,:,1])))
            if B2A.shape[1] >= 3:
                im_ech3 = np.squeeze(np.abs(tf.complex(B2A[:,2,:,:,0],B2A[:,2,:,:,1])))
            if B2A.shape[1] >= 4:
                im_ech4 = np.squeeze(np.abs(tf.complex(B2A[:,3,:,:,0],B2A[:,3,:,:,1])))
            if B2A.shape[1] >= 5:
                im_ech5 = np.squeeze(np.abs(tf.complex(B2A[:,4,:,:,0],B2A[:,4,:,:,1])))
            if B2A.shape[1] >= 6:
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
            if B2A.shape[1] >= 3:
                acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,2].set_title('3rd Echo')
                axs[0,2].axis('off')
            else:
                fig.delaxes(axs[0,2])
            if B2A.shape[1] >= 4:
                acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,3].set_title('4th Echo')
                axs[0,3].axis('off')
            else:
                fig.delaxes(axs[0,3])
            if B2A.shape[1] >= 5:
                acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,4].set_title('5th Echo')
                axs[0,4].axis('off')
            else:
                fig.delaxes(axs[0,4])
            if B2A.shape[1] >= 6:
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

            fn_aux = np.squeeze(B_WF_abs[:,1,:,:,:])
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
