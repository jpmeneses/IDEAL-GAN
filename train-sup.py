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

py.arg('--dataset', default='WF-sup')
py.arg('--data_size', type=int, default=192, choices=[192,384])
py.arg('--DL_gen', type=bool, default=False)
py.arg('--DL_experiment_dir', default='output/GAN-238')
py.arg('--n_per_epoch', type=int, default=10000)
py.arg('--n_echoes', type=int, default=6)
py.arg('--field', type=float, default=1.5)
py.arg('--out_vars', default='WF', choices=['WF','WFc','PM','WF-PM'])
py.arg('--G_model', default='multi-decod', choices=['multi-decod','U-Net','MEBCRN'])
py.arg('--n_G_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=8)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0005)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.9999)
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

if args.DL_gen:
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

dataset_hdf5_1 = 'INTA_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx,
                            acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

if not(args.DL_gen):
    dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

    dataset_hdf5_3 = 'Volunteers_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_3, out_maps_3 = data.load_hdf5(dataset_dir, dataset_hdf5_3, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

    dataset_hdf5_4 = 'Attilio_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_4, out_maps_4 = data.load_hdf5(dataset_dir, dataset_hdf5_4, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

################################################################################
############################# DATASET PARTITIONS ###############################
################################################################################

if args.DL_gen:
    trainX = np.zeros((args.n_per_epoch,1,1,1),dtype=np.float32)
    if args.G_model == 'MEBCRN':
        trainX = np.expand_dims(trainX, axis=-1)
else:
    trainX  = np.concatenate((acqs_2,acqs_3,acqs_4),axis=0)
valX = acqs_1

if args.DL_gen:
    trainY = np.zeros((args.n_per_epoch,1,1,1),dtype=np.float32)
    if args.G_model == 'MEBCRN':
        trainY = np.expand_dims(trainY, axis=-1)
else:
    trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4),axis=0)
valY = out_maps_1

# Overall dataset statistics
if args.G_model == 'MEBCRN':
    len_dataset,_,_,_,_ = np.shape(trainY)
    _,n_out,hgt,wdt,n_ch = np.shape(valY)
else:
    len_dataset,_,_,_ = np.shape(trainY)
    _,hgt,wdt,n_out = np.shape(valY)
    n_ch = 2

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:', args.n_echoes)
print('Output Maps:', n_out)

# Input and output dimensions (training data)
print('Training input shape:', trainX.shape)
print('Training output shape:', trainY.shape)

# Input and output dimensions (validations data)
print('Validation input shape:', valX.shape)
print('Validation output shape:', valY.shape)

A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
A_B_dataset = A_B_dataset.batch(args.batch_size).shuffle(len_dataset)
A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

if args.G_model == 'multi-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,args.n_echoes*2),
                                filters=args.n_G_filters,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,args.n_echoes*2),
                                ME_layer=False,
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WFc':
        n_out = 4
        out_activ = 'tanh'
    elif args.out_vars == 'WF-PM':
        n_out = 4
        out_activ = 'relu'
    else:
        n_out = 2
        out_activ = 'relu'
    G_A2B = dl.UNet(input_shape=(hgt,wdt,args.n_echoes*2),
                    n_out=n_out,
                    filters=args.n_G_filters,
                    output_activation=out_activ,
                    self_attention=args.D1_SelfAttention)
    if args.out_vars == 'WF':
        trainY[:,:,:,-1]    = 0.5*trainY[:,:,:,-1] + 0.5
        valY[:,:,:,-1]      = 0.5*valY[:,:,:,-1] + 0.5

elif args.G_model == 'MEBCRN':
    if args.out_vars=='WFc':
        n_out = 4
        out_activ = None
    else:
        n_out = 2
        out_activ = 'sigmoid'
    G_A2B=dl.MEBCRN(input_shape=(args.n_echoes,hgt,wdt,2),
                    n_outputs=n_out,
                    output_activation=out_activ,
                    n_res_blocks=9,
                    n_downsamplings=0,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

if args.DL_gen:
    if DL_args.div_decod:
        if DL_args.only_mag:
            nd = 2
        else:
            nd = 3
    else:
        nd = 1
    if len(DL_args.n_G_filt_list) == (DL_args.n_downsamplings+1):
        nfe = filt_list
        nfd = [a//nd for a in filt_list]
        nfd2 = [a//(nd+1) for a in filt_list]
    else:
        nfe = DL_args.n_G_filters
        nfd = DL_args.n_G_filters//nd
        nfd2= DL_args.n_G_filters//(nd+1)
    # enc= dl.encoder(input_shape=(None,hgt,wdt,n_ch),
    #                 encoded_dims=DL_args.encoded_size,
    #                 filters=nfe,
    #                 num_layers=DL_args.n_downsamplings,
    #                 num_res_blocks=DL_args.n_res_blocks,
    #                 NL_self_attention=DL_args.NL_SelfAttention
    #                 )
    if DL_args.only_mag:
        dec_mag = dl.decoder(encoded_dims=DL_args.encoded_size,
                            output_shape=(hgt,wdt,3),
                            filters=nfd,
                            num_layers=DL_args.n_downsamplings,
                            num_res_blocks=DL_args.n_res_blocks,
                            output_activation='relu',
                            output_initializer='he_normal',
                            NL_self_attention=DL_args.NL_SelfAttention
                            )
        dec_pha = dl.decoder(encoded_dims=DL_args.encoded_size,
                            output_shape=(hgt,wdt,2),
                            filters=nfd2,
                            num_layers=DL_args.n_downsamplings,
                            num_res_blocks=DL_args.n_res_blocks,
                            output_activation='tanh',
                            NL_self_attention=DL_args.NL_SelfAttention
                            )
        tl.Checkpoint(dict(dec_mag=dec_mag,dec_pha=dec_pha), py.join(args.DL_experiment_dir, 'checkpoints')).restore()
    else:
        dec_w =  dl.decoder(encoded_dims=DL_args.encoded_size,
                            output_shape=(hgt,wdt,n_ch),
                            filters=nfd,
                            num_layers=DL_args.n_downsamplings,
                            num_res_blocks=DL_args.n_res_blocks,
                            output_activation=None,
                            NL_self_attention=DL_args.NL_SelfAttention
                            )
        dec_f =  dl.decoder(encoded_dims=DL_args.encoded_size,
                            output_shape=(hgt,wdt,n_ch),
                            filters=nfd,
                            num_layers=DL_args.n_downsamplings,
                            num_res_blocks=DL_args.n_res_blocks,
                            output_activation=None,
                            NL_self_attention=DL_args.NL_SelfAttention
                            )
        dec_xi = dl.decoder(encoded_dims=DL_args.encoded_size,
                            output_shape=(hgt,wdt,n_ch),
                            n_groups=DL_args.n_groups_PM,
                            filters=nfd,
                            num_layers=args.n_downsamplings,
                            num_res_blocks=args.n_res_blocks,
                            output_activation=None,
                            NL_self_attention=args.NL_SelfAttention
                            )
        tl.Checkpoint(dict(dec_w=dec_w,dec_f=dec_f,dec_xi=dec_xi), py.join(args.DL_experiment_dir, 'checkpoints')).restore()

sup_loss_fn = tf.losses.MeanAbsoluteError()

if DL_args.only_mag:
    IDEAL_op = wf.IDEAL_mag_Layer()
else:
    IDEAL_op = wf.IDEAL_Layer()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    B_WF = B[:,:,:,:4]
    B_PM = B[:,:,:,4:]
    B_WF_abs = tf.abs(tf.complex(B_WF[:,:,:,0::2],B_WF[:,:,:,1::2]))
    with tf.GradientTape() as t:
        if args.out_vars == 'WF':
            # Compute model's output
            A2B_WF_abs = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_WF_abs = tf.where(B[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)

            # Compute zero-valued param maps
            A2B_PM = tf.zeros_like(A2B_WF_abs)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)

        elif args.out_vars == 'WFc':
            # Compute model's output
            A2B_WF = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_WF = tf.where(B[:,:,:,:4]!=0.0,A2B_WF,0.0)

            # Magnitude of water/fat images
            A2B_WF_real = A2B_WF[:,:,:,0::2]
            A2B_WF_imag = A2B_WF[:,:,:,1::2]
            A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))

            # Compute zero-valued param maps
            A2B_PM = tf.zeros_like(A2B_WF_abs)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF, A2B_WF)

        elif args.out_vars == 'PM':
            # Compute model's output
            A2B_PM = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_PM = tf.where(B[:,:,:,:2]!=0.0,A2B_PM,0.0)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(B[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)

            # Compute water/fat
            A2B_WF = wf.get_rho(A, A2B_PM, MEBCRN=False)
            
            # Magnitude of water/fat images
            A2B_WF_real = A2B_WF[:,:,:,0::2]
            A2B_WF_imag = A2B_WF[:,:,:,1::2]
            A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
            
            # Compute loss
            sup_loss = sup_loss_fn(B_PM, A2B_PM)

        elif args.out_vars == 'WF-PM':
            # Compute model's output
            B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
            A2B_abs = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_abs = tf.where(B[:,:,:,:4]!=0.0,A2B_abs,0.0)

            # Split A2B outputs
            A2B_WF_abs = A2B_abs[:,:,:,:2]
            A2B_PM = A2B_abs[:,:,:,2:]

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(B[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)

            # Compute loss
            sup_loss = sup_loss_fn(B_abs, A2B_abs)

        ############### Splited losses ####################
        WF_abs_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
        R2_loss = sup_loss_fn(B_PM[:,:,:,:1], A2B_R2)
        FM_loss = sup_loss_fn(B_PM[:,:,:,1:], A2B_FM)

        ################ Regularizers #####################
        if not(args.out_vars=='WF' or args.out_vars=='WFc'):
            R2_TV = tf.reduce_sum(tf.image.total_variation(A2B_R2)) * args.R2_TV_weight
            FM_TV = tf.reduce_sum(tf.image.total_variation(A2B_FM)) * args.FM_TV_weight
            R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_R2),axis=(1,2,3))) * args.R2_L1_weight
            FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_FM),axis=(1,2,3))) * args.FM_L1_weight
        else:
            R2_TV = 0.0
            FM_TV = 0.0
            R2_L1 = 0.0
            FM_L1 = 0.0
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


def train_step(A, B):
    G_loss_dict = train_G(A, B)
    return G_loss_dict


@tf.function
def sample(A, B):
    B_WF = B[:,:,:,:4]
    B_PM = B[:,:,:,4:]
    B_WF_abs = tf.abs(tf.complex(B_WF[:,:,:,0::2],B_WF[:,:,:,1::2]))
    # Estimate A2B
    if args.out_vars == 'WF':
        A2B_WF_abs = G_A2B(A, training=True)
        A2B_WF_abs = tf.where(B[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)
        A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
    elif args.out_vars == 'WFc':
        A2B_WF = G_A2B(A, training=True)
        A2B_WF = tf.where(B[:,:,:,:4]!=0.0,A2B_WF,0.0)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF, A2B_WF)
    elif args.out_vars == 'PM':
        A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(B_PM!=0.0,A2B_PM,0.0)
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM,MEBCRN=False)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_PM, A2B_PM)
    elif args.out_vars == 'WF-PM':
        B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
        A2B_abs = G_A2B(A, training=True)
        A2B_abs = tf.where(B_abs!=0.0,A2B_abs,0.0)
        # Split A2B outputs
        A2B_WF_abs = A2B_abs[:,:,:,:2]
        A2B_PM = A2B_abs[:,:,:,2:]
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)
        val_sup_loss = sup_loss_fn(B_abs, A2B_abs)

    ############### Splited losses ####################
    WF_abs_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
    R2_loss = sup_loss_fn(B_PM[:,:,:,:1], A2B_R2)
    FM_loss = sup_loss_fn(B_PM[:,:,:,1:], A2B_FM)

    return A2B_abs,{'sup_loss': val_sup_loss,
                    'WF_loss': WF_abs_loss,
                    'R2_loss': R2_loss,
                    'FM_loss': FM_loss}

def validation_step(A, B):
    A2B_abs, val_sup_dict = sample(A, B)
    return A2B_abs, val_sup_dict

if args.DL_gen:
    @tf.function
    def gen_sample(Z,TE=None):
        # Z2B2A Cycle
        if DL_args.only_mag:
            Z2B_mag = dec_mag(Z, training=True)
            Z2B_pha = dec_pha(Z, training=True)
            Z2B_pha = tf.concat([tf.zeros_like(Z2B_pha[:,:,:,:,:1]),Z2B_pha],axis=-1)
            Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
        else:
            Z2B_w = dec_w(Z, training=False)
            Z2B_f = dec_f(Z, training=False)
            Z2B_xi= dec_xi(Z, training=False)
            Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
        # Calculate CSE-MRI data (in non-MEBCRN format)
        Z2B2A = IDEAL_op(Z2B)
        # rho_hat = tf.transpose(rho_hat, perm=[0,2,3,1])
        Re_rho = tf.transpose(Z2B2A[:,:,:,:,0], perm=[0,2,3,1])
        Im_rho = tf.transpose(Z2B2A[:,:,:,:,1], perm=[0,2,3,1])
        zero_fill = tf.zeros_like(Re_rho)
        re_stack = tf.stack([Re_rho,zero_fill],4)
        re_aux = tf.reshape(re_stack,[Z.shape[0],hgt,wdt,2*args.n_echoes])
        im_stack = tf.stack([zero_fill,Im_rho],4)
        im_aux = tf.reshape(im_stack,[Z.shape[0],hgt,wdt,2*args.n_echoes])
        Z2B2A = re_aux + im_aux
        # Turn Z2B into non-MEBCRN format
        if DL_args.only_mag:
            Z2B_W_r = Z2B_mag[:,0,:,:,:1] * tf.math.cos(Z2B_pha[:,0,:,:,1:2]*np.pi)
            Z2B_W_i = Z2B_mag[:,0,:,:,:1] * tf.math.sin(Z2B_pha[:,0,:,:,1:2]*np.pi)
            Z2B_F_r = Z2B_mag[:,0,:,:,1:2]* tf.math.cos(Z2B_pha[:,0,:,:,1:2]*np.pi)
            Z2B_F_i = Z2B_mag[:,0,:,:,1:2]* tf.math.sin(Z2B_pha[:,0,:,:,1:2]*np.pi)
            Z2B_r2 = Z2B_mag[:,0,:,:,2:]
            Z2B_fm = Z2B_pha[:,0,:,:,2:]
            Z2B = tf.concat([Z2B_W_r,Z2B_W_i,Z2B_F_r,Z2B_F_i,Z2B_r2,Z2B_fm],axis=-1)
        else:
            Z2B= tf.concat([tf.squeeze(Z2B_w,axis=1),
                            tf.squeeze(Z2B_f,axis=1),
                            tf.squeeze(Z2B_xi,axis=1)],axis=-1)
        return Z2B, Z2B2A


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
n_div = np.ceil(total_steps/len(valX))

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
            z_shape = (A.shape[0],hls,wls,DL_args.encoded_size)
            Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
            B, A = gen_sample(Z)
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        # p = np.random.rand()
        # if p <= 0.4:
        #     # Random 90 deg rotations
        #     for _ in range(np.random.randint(3)):
        #         B = tf.image.rot90(B)

        #     # Random horizontal reflections
        #     B = tf.image.random_flip_left_right(B)

        #     # Random vertical reflections
        #     B = tf.image.random_flip_up_down(B)
        # ==============================================================================
        
        G_loss_dict = train_step(A, B)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200):
            A, B = next(val_iter)
            # A = A[0,...]
            # B = B[0,...]

            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A2B, val_sup_dict = validation_step(A, B)

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_sup_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            if args.G_model == 'MEBCRN':
                im_ech1 = np.squeeze(np.abs(tf.complex(A[:,0,:,:,0],A[:,0,:,:,1])))
                im_ech2 = np.squeeze(np.abs(tf.complex(A[:,1,:,:,0],A[:,1,:,:,1])))
                if args.n_echoes >= 3:
                    im_ech3 = np.squeeze(np.abs(tf.complex(A[:,2,:,:,0],A[:,2,:,:,1])))
                if args.n_echoes >= 4:
                    im_ech4 = np.squeeze(np.abs(tf.complex(A[:,3,:,:,0],A[:,3,:,:,1])))
                if args.n_echoes >= 5:
                    im_ech5 = np.squeeze(np.abs(tf.complex(A[:,4,:,:,0],A[:,4,:,:,1])))
                if args.n_echoes >= 6:
                    im_ech6 = np.squeeze(np.abs(tf.complex(A[:,5,:,:,0],A[:,5,:,:,1])))
            else:
                im_ech1 = np.squeeze(np.abs(tf.complex(A[:,:,:,0],A[:,:,:,1])))
                im_ech2 = np.squeeze(np.abs(tf.complex(A[:,:,:,2],A[:,:,:,3])))
                if args.n_echoes >= 3:
                    im_ech3 = np.squeeze(np.abs(tf.complex(A[:,:,:,4],A[:,:,:,5])))
                if args.n_echoes >= 4:
                    im_ech4 = np.squeeze(np.abs(tf.complex(A[:,:,:,6],A[:,:,:,7])))
                if args.n_echoes >= 5:
                    im_ech5 = np.squeeze(np.abs(tf.complex(A[:,:,:,8],A[:,:,:,9])))
                if args.n_echoes >= 6:
                    im_ech6 = np.squeeze(np.abs(tf.complex(A[:,:,:,10],A[:,:,:,11])))

            
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

            # A2B maps in the second row
            w_aux = np.squeeze(A2B[:,:,:,0])
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(A2B[:,:,:,1])
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(A2B[:,:,:,2])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(A2B[:,:,:,3])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,4])
            axs[1,4].axis('off')
            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            wn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,0],B[:,:,:,1])))
            W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            fn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,2],B[:,:,:,3])))
            F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            r2n_aux = np.squeeze(B[:,:,:,4])
            r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            fieldn_aux = np.squeeze(B[:,:,:,5])
            field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_unet, ax=axs[2,4])
            axs[2,4].axis('off')
            fig.delaxes(axs[2,0])
            fig.delaxes(axs[2,5])

            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
