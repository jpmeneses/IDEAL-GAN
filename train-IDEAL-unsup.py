import functools

import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
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
py.arg('--rand_ne', type=bool, default=False)
py.arg('--out_vars', default='FM', choices=['R2s','FM','PM'])
py.arg('--UQ', type=bool, default=False)
py.arg('--ME_layer', type=bool, default=True)
py.arg('--k_fold', type=int, default=1)
py.arg('--n_G_filters', type=int, default=32)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=60)
py.arg('--epoch_decay', type=int, default=60)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=5)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0001)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--data_aug_p', type=float, default=0.4)
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--R2_L1_weight', type=float, default=0.0)
py.arg('--FM_TV_weight', type=float, default=0.0)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=True)
py.arg('--D2_SelfAttention',type=bool, default=False)
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

fm_sc = 300.0
r2_sc = 200.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_384_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, end=100,
                            acqs_data=True, te_data=False, MEBCRN=True)

dataset_hdf5_2 = 'INTA_GC_384_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2,
                            acqs_data=True, te_data=False, MEBCRN=True)

dataset_hdf5_3 = 'INTArest_GC_384_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3,
                            acqs_data=True, te_data=False, MEBCRN=True)

dataset_hdf5_4 = 'Volunteers_GC_384_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4,
                            acqs_data=True, te_data=False, MEBCRN=True)

dataset_hdf5_5 = 'Attilio_GC_384_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5,
                            acqs_data=True, te_data=False, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX = np.concatenate((acqs_1,acqs_2,acqs_3,acqs_4,acqs_5),axis=0)
trainY = np.concatenate((out_maps_1,out_maps_2,out_maps_3,out_maps_4,out_maps_5),axis=0)
k_divs = [0,832,1694,2547,3409,len(trainX)]

valX = trainX[k_divs[args.k_fold-1]:k_divs[args.k_fold],:,:,:,:]
valY = trainY[k_divs[args.k_fold-1]:k_divs[args.k_fold],:,:,:,:]

trainX = np.delete(trainX,np.s_[k_divs[args.k_fold-1]:k_divs[args.k_fold]],0)
trainY = np.delete(trainY,np.s_[k_divs[args.k_fold-1]:k_divs[args.k_fold]],0)

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(trainX)
_,n_out,_,_,_ = np.shape(valY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:', ne)
print('Output Maps:', n_out)

# Input and output dimensions (validations data)
print('Training output shape:', trainY.shape)
print('Validation output shape:', valY.shape)

A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
A_B_dataset = A_B_dataset.batch(args.batch_size).shuffle(len_dataset)
A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

G_A2B = dl.UNet(input_shape=(None,hgt,wdt,n_ch),
                bayesian=args.UQ,
                ME_layer=args.ME_layer,
                filters=args.n_G_filters,
                self_attention=args.D1_SelfAttention)
if args.out_vars == 'R2s' or args.out_vars == 'PM':
    G_A2R2= dl.UNet(input_shape=(None,hgt,wdt,1),
                    bayesian=args.UQ,
                    ME_layer=args.ME_layer,
                    filters=args.n_G_filters,
                    output_activation='sigmoid',
                    output_initializer='he_uniform',
                    self_attention=args.D2_SelfAttention)

cycle_loss_fn = tf.losses.MeanSquaredError()
uncertain_loss = gan.VarMeanSquaredError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)
if not(args.out_vars == 'FM'):
    G_R2_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        ##################### A Cycle #####################
        if args.UQ:
            A2B_FM, _, A2B_FM_var = G_A2B(A, training=True) # Randomly sampled FM
        else:
            A2B_FM = G_A2B(A, training=True)
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)

        if args.out_vars == 'PM':
            A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
            # Compute R2s map from only-mag images
            if args.UQ:
                A2B_R2, A2B_R2_nu, A2B_R2_sigma = G_A2R2(A_abs, training=(args.out_vars=='PM')) # Randomly sampled R2s
            else:
                A2B_R2 = G_A2R2(A_abs, training=(args.out_vars=='PM'))
                # A2B_R2_var = None
            A2B_R2 = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2,0.0)

        else:
            A2B_R2 = tf.zeros_like(A2B_FM)
            if args.UQ:
                A2B_R2_nu = tf.zeros_like(A2B_FM_var)
                A2B_R2_sigma = tf.zeros_like(A2B_FM_var)

        A2B_PM = tf.concat([A2B_FM,A2B_R2], axis=-1)

        # Magnitude of water/fat images
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))
        A2B = tf.concat([A2B_WF,A2B_PM], axis=1)

        # Variance map mask and attach to recon-A
        if args.UQ:
            A2B_FM_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM_var,0.0)
            A2B_R2_nu = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_nu,0.0)
            A2B_R2_sigma = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_sigma,0.0)
            A2B_PM_var = tf.concat([A2B_FM_var,A2B_R2_nu,A2B_R2_sigma], axis=-1)
            A2B2A_var = wf.acq_uncertainty(A2B, A2B_PM_var, ne=A.shape[1], rem_R2=(args.out_vars=='FM'))
            A2B2A_sampled_var = tf.concat([A2B2A, A2B2A_var], axis=-1) # shape: [nb,ne,hgt,wdt,4]

        ############ Cycle-Consistency Losses #############
        if args.UQ:
            A2B2A_cycle_loss = uncertain_loss(A, A2B2A_sampled_var)
        else:
            A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)

        ########### Splitted R2s and FM Losses ############
        B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
        WF_abs_loss = cycle_loss_fn(B_WF_abs, A2B_WF_abs)
        R2_loss = cycle_loss_fn(B[:,2:,:,:,1:], A2B_R2)
        FM_loss = cycle_loss_fn(B[:,2:,:,:,:1], A2B_FM)

        ################ Regularizers #####################
        FM_TV = tf.reduce_sum(tf.image.total_variation(A2B_FM[:,0,:,:,:]))
        FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_FM),axis=(1,2,3,4)))
        reg_term = FM_TV * args.FM_TV_weight + FM_L1 * args.FM_L1_weight
        
        G_loss = A2B2A_cycle_loss + reg_term
        
    if args.out_vars == 'FM':
        G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))
    else:
        G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_A2R2.trainable_variables)
        G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_A2R2.trainable_variables))
    
    return {'A2B2A_cycle_loss': A2B2A_cycle_loss,
            'WF_loss': WF_abs_loss,
            'R2_loss': R2_loss,
            'FM_loss': FM_loss,
            'TV_FM': FM_TV,
            'L1_FM': FM_L1}


@tf.function
def train_G_R2(A, B):
    with tf.GradientTape() as t:
        ##################### A Cycle #####################
        A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))

        # Compute R2s map from only-mag images
        if args.UQ:
            A2B_R2, A2B_R2_nu, A2B_R2_sigma = G_A2R2(A_abs, training=True) # Randomly sampled R2s
        else:
            A2B_R2 = G_A2R2(A_abs, training=True)
        A2B_R2 = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2,0.0)

        # Compute FM using complex-valued images and pre-trained model
        if args.UQ:
            A2B_FM, A2B_FM_var = G_A2B(A, training=False) # Mean FM
        else:
            A2B_FM = G_A2B(A, training=False)
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)
        A2B_PM = tf.concat([A2B_FM,A2B_R2], axis=-1)

        # Magnitude of water/fat images
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))
        A2B2A_abs = wf.IDEAL_mag(A2B_WF_abs, A2B_PM, ne=A.shape[1])
        
        # Variance map mask and attach to recon-A
        if args.UQ:
            A2B_FM_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM_var,0.0)
            A2B_R2_nu = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_nu,0.0)
            A2B_R2_sigma = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_sigma,0.0)
            A2B_PM_var = tf.concat([A2B_FM_var,A2B_R2_nu,A2B_R2_sigma], axis=-1)
            A2B2A_var = wf.acq_uncertainty(A2B, A2B_PM_var, ne=A.shape[1], only_mag=True)
            A2B2A_sampled_var = tf.concat([A2B2A_abs, A2B2A_var], axis=-1) # shape: [nb,ne,hgt,wdt,2]

        ############ Cycle-Consistency Losses #############
        # CHECK
        if args.UQ:
            A2B2A_cycle_loss = uncertain_loss(A_abs, A2B2A_sampled_var)
        else:
            A2B2A_cycle_loss = cycle_loss_fn(A_abs, A2B2A_abs)

        ########### Splitted R2s and FM Losses ############
        B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
        WF_abs_loss = cycle_loss_fn(B_WF_abs, A2B_WF_abs)
        R2_loss = cycle_loss_fn(B[:,2:,:,:,1:], A2B_R2)
        FM_loss = cycle_loss_fn(B[:,2:,:,:,:1], A2B_FM)

        ################ Regularizers #####################
        R2_TV = tf.reduce_sum(tf.image.total_variation(A2B_R2[:,0,:,:,:]))
        R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_R2),axis=(1,2,3,4)))
        reg_term = R2_TV * args.R2_TV_weight + R2_L1 * args.R2_L1_weight
        
        G_loss = A2B2A_cycle_loss + reg_term
        
    G_grad = t.gradient(G_loss, G_A2R2.trainable_variables)
    G_R2_optimizer.apply_gradients(zip(G_grad, G_A2R2.trainable_variables))

    return {'A2B2A_cycle_loss': A2B2A_cycle_loss,
            'WF_loss': WF_abs_loss,
            'R2_loss': R2_loss,
            'FM_loss': FM_loss,
            'TV_R2': R2_TV,
            'L1_R2': R2_L1}


def train_step(A, B):
    if args.out_vars == 'R2s':
        G_loss_dict  = {'A2B2A_cycle_loss': tf.constant(0.0),
                        'WF_loss': tf.constant(0.0),
                        'R2_loss': tf.constant(0.0),
                        'FM_loss': tf.constant(0.0),
                        'TV_FM': tf.constant(0.0),
                        'L1_FM': tf.constant(0.0)}
        G_R2_loss_dict = train_G_R2(A, B)
    else:
        G_loss_dict = train_G(A, B)
        G_R2_loss_dict={'A2B2A_cycle_loss': tf.constant(0.0),
                        'WF_loss': tf.constant(0.0),
                        'R2_loss': tf.constant(0.0),
                        'FM_loss': tf.constant(0.0),
                        'TV_R2': tf.constant(0.0),
                        'L1_R2': tf.constant(0.0)}
    return G_loss_dict, G_R2_loss_dict


@tf.function
def sample(A, B):
    if args.out_vars == 'FM':
        if args.UQ:
            A2B_FM, _, A2B_FM_var = G_A2B(A, training=False)
            A2B_R2_nu = tf.zeros_like(A2B_FM_var)
            A2B_R2_sigma = tf.zeros_like(A2B_FM_var)
        else:
            A2B_FM = G_A2B(A, training=False)
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)

        # Build A2B_PM array with zero-valued R2*
        A2B_R2 = tf.zeros_like(A2B_FM)
        A2B_PM = tf.concat([A2B_FM,A2B_R2], axis=-1)
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B = tf.concat([A2B_WF, A2B_PM],axis=1)

        # Magnitude of water/fat images
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))

    elif args.out_vars == 'R2s' or args.out_vars == 'PM':
        A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
        # Compute FM from complex-valued images
        if args.UQ:
            A2B_FM, A2B_FM_var = G_A2B(A, training=False)
            A2B_R2, A2B_R2_nu, A2B_R2_sigma = G_A2R2(A_abs, training=False)
        else:
            A2B_FM = G_A2B(A, training=False)
            A2B_R2 = G_A2R2(A_abs, training=True)
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)

        # Compute R2s maps using only-mag images
        A2B_R2 = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2,0.0)
        A2B_PM = tf.concat([A2B_FM,A2B_R2], axis=-1)

        # Magnitude of water/fat images
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B = tf.concat([A2B_WF,A2B_PM], axis=1)
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))
        A2B2A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B2A),axis=-1,keepdims=True))
    
    # Variance map mask and attach to recon-A
    if args.UQ:
        A2B_FM_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM_var,0.0)
        A2B_R2_nu = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_nu,0.0)
        A2B_R2_sigma = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_sigma,0.0)
        A2B_PM_var = tf.concat([A2B_FM_var,A2B_R2_nu,A2B_R2_sigma], axis=-1)
        A2B2A_var = wf.acq_uncertainty(A2B, A2B_PM_var, ne=A.shape[1])
        A2B2A_sampled_var = tf.concat([A2B2A, A2B2A_var], axis=-1) # shape: [nb,ne,hgt,wdt,4]
    else:
        A2B_PM_var = None

    ########### Splitted R2s and FM Losses ############
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,:,:,:]),axis=-1,keepdims=True))
    WF_abs_loss = cycle_loss_fn(B_WF_abs, A2B_WF_abs)
    R2_loss = cycle_loss_fn(B[:,2:,:,:,1:], A2B_R2)
    FM_loss = cycle_loss_fn(B[:,2:,:,:,:1], A2B_FM)

    if args.UQ:
        if args.out_vars == 'FM':
            val_A2B2A_R2_loss = 0
            val_A2B2A_FM_loss = uncertain_loss(A, A2B2A_sampled_var)
        else:
            # val_A2B2A_R2_loss = uncertain_loss(A_abs, A2B2A_abs_sampled_var)
            val_A2B2A_R2_loss = 0
            val_A2B2A_FM_loss = uncertain_loss(A, A2B2A_sampled_var)
    else:
        if args.out_vars == 'FM':
            val_A2B2A_R2_loss = 0
            val_A2B2A_FM_loss = cycle_loss_fn(A, A2B2A)
        else:
            val_A2B2A_R2_loss = cycle_loss_fn(A_abs, A2B2A_abs)
            val_A2B2A_FM_loss = cycle_loss_fn(A, A2B2A)
    
    val_FM_dict =  {'A2B2A_cycle_loss': val_A2B2A_FM_loss,
                    'WF_loss': WF_abs_loss,
                    'R2_loss': R2_loss,
                    'FM_loss': FM_loss}
    val_R2_dict =  {'A2B2A_cycle_loss': val_A2B2A_R2_loss,
                    'WF_loss': WF_abs_loss,
                    'R2_loss': R2_loss,
                    'FM_loss': FM_loss}

    return A2B, A2B_PM_var, val_FM_dict, val_R2_dict

def validation_step(A, B):
    A2B, A2B_var, val_FM_dict, val_R2_dict = sample(A, B)
    return A2B, A2B_var, val_FM_dict, val_R2_dict


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
if not(args.out_vars == 'FM'):
    checkpoint_2=tl.Checkpoint(dict(G_A2B=G_A2B,
                                    G_A2R2=G_A2R2,
                                    G_optimizer=G_optimizer,
                                    G_R2_optimizer=G_R2_optimizer,
                                    ep_cnt=ep_cnt),
                               py.join(output_dir, 'checkpoints'),
                               max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    if args.out_vars == 'PM':
        checkpoint_2.restore().assert_existing_objects_matched()
    else:
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
n_div = np.ceil(total_steps/len(valY))//10

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for A, B in A_B_dataset:
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        p = np.random.rand()
        if p <= args.data_aug_p:
            aux_ne = A.shape[1]
            A = tf.reshape(tf.transpose(A,perm=[0,2,3,1,4]),[A.shape[0],hgt,wdt,aux_ne*n_ch])
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

            A = tf.transpose(tf.reshape(A,[A.shape[0],hgt,wdt,aux_ne,n_ch]),[0,3,1,2,4])
            B = tf.transpose(tf.reshape(B,[B.shape[0],hgt,wdt,n_out,n_ch]),[0,3,1,2,4])
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        if args.rand_ne:
            ne_sel = np.random.randint(3,7)
            A = A[:,:ne_sel,:,:,:]

        G_loss_dict, G_R2_loss_dict = train_step(A, B)

        if args.out_vars == 'R2s':
            opt_aux = G_R2_optimizer.iterations
        else:
            opt_aux = G_optimizer.iterations

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=opt_aux, name='G_losses')
            tl.summary(G_R2_loss_dict, step=opt_aux, name='G_R2_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=opt_aux, name='G learning rate')

        # sample
        if (opt_aux.numpy() % n_div == 0) or (opt_aux.numpy() < 200):
            A, B = next(val_iter)
            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A = A[:,:ne_sel,:,:,:]
            A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=False))
            A2B, A2B_var, val_FM_dict, val_R2_dict = validation_step(A, B)

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_FM_dict, step=opt_aux, name='G_losses')
                tl.summary(val_R2_dict, step=opt_aux, name='G_R2_losses')

            if (opt_aux.numpy() % (n_div*100) == 0) or (opt_aux.numpy() < 100):
                fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

                # Magnitude of recon MR images at each echo
                im_ech1 = np.squeeze(A_abs[:,0,:,:])
                im_ech2 = np.squeeze(A_abs[:,1,:,:])
                if A.shape[1] >= 3:
                    im_ech3 = np.squeeze(A_abs[:,2,:,:])
                if A.shape[1] >= 4:
                    im_ech4 = np.squeeze(A_abs[:,3,:,:])
                if A.shape[1] >= 5:
                    im_ech5 = np.squeeze(A_abs[:,4,:,:])
                if A.shape[1] >= 6:
                    im_ech6 = np.squeeze(A_abs[:,5,:,:])
                
                # Acquisitions in the first row
                acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,0].set_title('1st Echo')
                axs[0,0].axis('off')
                acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,1].set_title('2nd Echo')
                axs[0,1].axis('off')
                if A.shape[1] >= 3:
                    acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                          interpolation='none', vmin=0, vmax=1)
                    axs[0,2].set_title('3rd Echo')
                    axs[0,2].axis('off')
                else:
                    fig.delaxes(axs[0,2])
                if A.shape[1] >= 4:
                    acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                          interpolation='none', vmin=0, vmax=1)
                    axs[0,3].set_title('4th Echo')
                    axs[0,3].axis('off')
                else:
                    fig.delaxes(axs[0,3])
                if A.shape[1] >= 5:
                    acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                          interpolation='none', vmin=0, vmax=1)
                    axs[0,4].set_title('5th Echo')
                    axs[0,4].axis('off')
                else:
                    fig.delaxes(axs[0,4])
                if A.shape[1] >= 6:
                    acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                          interpolation='none', vmin=0, vmax=1)
                    axs[0,5].set_title('6th Echo')
                    axs[0,5].axis('off')
                else:
                    fig.delaxes(axs[0,5])

                # A2B maps in the second row
                w_aux = np.squeeze(tf.math.sqrt(tf.reduce_sum(tf.square(A2B[:,0,:,:,:]),axis=-1,keepdims=False)))
                W_ok =  axs[1,0].imshow(w_aux, cmap='bone',
                                        interpolation='none', vmin=0, vmax=1)
                fig.colorbar(W_ok, ax=axs[1,0])
                axs[1,0].axis('off')

                f_aux = np.squeeze(tf.math.sqrt(tf.reduce_sum(tf.square(A2B[:,1,:,:,:]),axis=-1,keepdims=False)))
                F_ok =  axs[1,1].imshow(f_aux, cmap='pink',
                                        interpolation='none', vmin=0, vmax=1)
                fig.colorbar(F_ok, ax=axs[1,1])
                axs[1,1].axis('off')

                r2_aux = np.squeeze(A2B[:,2,:,:,1])
                r2_ok = axs[1,2].imshow(r2_aux*r2_sc, cmap='copper',
                                        interpolation='none', vmin=0, vmax=r2_sc)
                fig.colorbar(r2_ok, ax=axs[1,2])
                axs[1,2].axis('off')

                field_aux = np.squeeze(A2B[:,2,:,:,0])
                field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                            interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
                fig.colorbar(field_ok, ax=axs[1,4])
                axs[1,4].axis('off')
                
                if args.UQ:
                    R2_var_aux = np.squeeze(A2B_var[:,0,:,:,1])*(r2_sc**2)
                    R2_var_ok= axs[1,3].imshow(R2_var_aux, cmap='gnuplot',
                                            interpolation='none', vmin=0, vmax=5)
                    fig.colorbar(R2_var_ok, ax=axs[1,3])
                    axs[1,3].axis('off')

                    FM_var_aux = np.squeeze(A2B_var[:,0,:,:,0])*(fm_sc**2)
                    FM_var_ok= axs[1,5].imshow(FM_var_aux, cmap='gnuplot2',
                                            interpolation='none', vmin=0, vmax=5)
                    fig.colorbar(FM_var_ok, ax=axs[1,5])
                    axs[1,5].axis('off')
                else:
                    fig.delaxes(axs[1,3])
                    fig.delaxes(axs[1,5])

                # Ground-truth in the third row
                wn_aux = np.squeeze(tf.math.sqrt(tf.reduce_sum(tf.square(B[:,0,:,:,:]),axis=-1,keepdims=False)))
                W_unet = axs[2,0].imshow(wn_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
                fig.colorbar(W_unet, ax=axs[2,0])
                axs[2,0].axis('off')

                fn_aux = np.squeeze(tf.math.sqrt(tf.reduce_sum(tf.square(B[:,1,:,:,:]),axis=-1,keepdims=False)))
                F_unet = axs[2,1].imshow(fn_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
                fig.colorbar(F_unet, ax=axs[2,1])
                axs[2,1].axis('off')

                r2n_aux = np.squeeze(B[:,2,:,:,1])
                r2_unet = axs[2,2].imshow(r2n_aux*r2_sc, cmap='copper',
                                     interpolation='none', vmin=0, vmax=r2_sc)
                fig.colorbar(r2_unet, ax=axs[2,2])
                axs[2,2].axis('off')

                fieldn_aux = np.squeeze(B[:,2,:,:,0])
                field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
                fig.colorbar(field_unet, ax=axs[2,4])
                axs[2,4].axis('off')
                fig.delaxes(axs[2,3])
                fig.delaxes(axs[2,5])

                if args.out_vars == 'R2s':
                    fig.suptitle('A2B Error: '+str(val_R2_dict['WF_loss']), fontsize=16)
                else:
                    fig.suptitle('A2B Error: '+str(val_FM_dict['WF_loss']), fontsize=16)

                # plt.show()
                plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
                tl.make_space_above(axs,topmargin=0.8)
                plt.savefig(py.join(sample_dir, 'iter-%09d.png' % opt_aux.numpy()),
                            bbox_inches = 'tight', pad_inches = 0)
                plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        if args.out_vars == 'FM':
            checkpoint.save(ep)
        else:
            checkpoint_2.save(ep)
