import functools

import random
import numpy as np
import matplotlib.pyplot as plt

import DLlib as dl
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import wflib as wf

import data
from keras_unet.models import custom_unet

from itertools import cycle

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='WF-sup')
py.arg('--n_echoes', type=int, default=6)
py.arg('--out_vars', default='WF', choices=['WF','PM','WF-PM'])
py.arg('--G_model', default='encod-decod', choices=['multi-decod','U-Net','MEBCRN'])
py.arg('--n_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=32)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.001)
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
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

################################################################################
############################# DATASET PARTITIONS ###############################
################################################################################

n1_div = 248
n3_div = 0
n4_div = 434

trainX  = np.concatenate((acqs_1[n1_div:,:,:,:],acqs_3,acqs_4[n4_div:,:,:,:],acqs_5),axis=0)
valX    = acqs_2
testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)

trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3[n3_div:,:,:,:],out_maps_4[n4_div:,:,:,:],out_maps_5),axis=0)
valY    = out_maps_2
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(trainX)
_,_,_,n_out = np.shape(trainY)
echoes = int(d_ech/2)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (training data)
print('Training input shape:',trainX.shape)
print('Training output shape:',trainY.shape)

# Input and output dimensions (validations data)
print('Validation input shape:',valX.shape)
print('Validation output shape:',valY.shape)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

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
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,d_ech),
                                filters=args.n_filters,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                                filters=args.n_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        n_out = 4
    else:
        n_out = 2
    G_A2B = custom_unet(input_shape=(hgt,wdt,d_ech),
                        num_classes=n_out,
                        dropout=0,
                        use_attention=args.D1_SelfAttention,
                        filters=args.n_filters)
    if not(args.out_vars == 'WF'):
        trainY[:,:,:,-1]    = 0.5*trainY[:,:,:,-1] + 0.5
        valY[:,:,:,-1]      = 0.5*valY[:,:,:,-1] + 0.5
        testY[:,:,:,-1]     = 0.5*testY[:,:,:,-1] + 0.5

elif args.G_model == 'MEBCRN':
    if args.out_vars == 'WF-PM':
        n_out = 4
    else:
        n_out = 2
    G_A2B=dl.MEBCRN(input_shape=(hgt,wdt,d_ech),
                    n_outputs=n_out,
                    n_res_blocks=5,
                    n_downsamplings=2,
                    filters=args.n_filters,
                    self_attention=args.D1_SelfAttention)
    if not(args.out_vars == 'WF'):
        trainY[:,:,:,-1]    = 0.5*trainY[:,:,:,-1] + 0.5
        valY[:,:,:,-1]      = 0.5*valY[:,:,:,-1] + 0.5
        testY[:,:,:,-1]     = 0.5*testY[:,:,:,-1] + 0.5

else:
    raise(NameError('Unrecognized Generator Architecture'))

sup_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    indx_B = tf.concat([tf.zeros_like(B[:,:,:,:2],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:2],dtype=tf.int32)],axis=-1)

    indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)

    with tf.GradientTape() as t:
        # Split B outputs
        B_WF,B_PM = tf.dynamic_partition(B,indx_B,num_partitions=2)
        B_WF = tf.reshape(B_WF,B[:,:,:,:2].shape)
        B_PM = tf.reshape(B_PM,B[:,:,:,4:].shape)

        # Magnitude of water/fat images
        B_WF_real = B_WF[:,:,:,0::2]
        B_WF_imag = B_WF[:,:,:,1::2]
        B_WF_abs = tf.abs(tf.complex(B_WF_real,B_WF_imag))

        if args.out_vars == 'WF':
            # Compute model's output
            A2B_WF_abs = G_A2B(A, training=True)
            A2B_WF_abs = tf.where(A[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)

            # Compute loss
            sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)

        elif args.out_vars == 'PM':
            # Compute model's output
            A2B_PM = G_A2B(A, training=True)
            A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0.0)

            # Split A2B param maps
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)

            # Compute water/fat
            A2B_WF = wf.get_rho(A, A2B_PM)
            
            # Magnitude of water/fat images
            A2B_WF_real = A2B_WF[:,:,:,0::2]
            A2B_WF_imag = A2B_WF[:,:,:,1::2]
            A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
            
            # Compute loss
            sup_loss = sup_loss_fn(B_PM, A2B_PM)

        elif args.out_vars == 'WF-PM':
            # Compute model's output
            A2B_abs = G_A2B(A, training=True)
            A2B_abs = tf.where(A[:,:,:,:4]!=0.0,A2B_abs,0.0)

            # Split A2B outputs
            A2B_WF_abs, A2B_PM = tf.dynamic_partition(A2B_abs,indx_B,num_partitions=2)
            A2B_WF_abs = tf.reshape(A2B_WF_abs,B[:,:,:,:2].shape)
            A2B_PM = tf.reshape(A2B_PM,B[:,:,:,4:].shape)

            # Split A2B param maps
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)

            # Compute loss
            B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
            sup_loss = sup_loss_fn(B_abs, A2B_abs)

        ################ Regularizers #####################
        if not(args.out_vars=='WF'):
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
            'TV_R2': R2_TV,
            'TV_FM': FM_TV,
            'L1_R2': R2_L1,
            'L1_FM': FM_L1}


def train_step(A, B):
    G_loss_dict = train_G(A, B)
    return G_loss_dict


@tf.function
def sample(A, B):
    indx_B = tf.concat([tf.zeros_like(B[:,:,:,:2],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:2],dtype=tf.int32)],axis=-1)
    indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
    # Split B
    B_WF,B_PM = tf.dynamic_partition(B,indx_B,num_partitions=2)
    B_WF = tf.reshape(B_WF,B[:,:,:,:2].shape)
    B_PM = tf.reshape(B_PM,B[:,:,:,4:].shape)
    # Magnitude of water/fat images
    B_WF_real = B_WF[:,:,:,0::2]
    B_WF_imag = B_WF[:,:,:,1::2]
    B_WF_abs = tf.abs(tf.complex(B_WF_real,B_WF_imag))
    # Estimate A2B
    if args.out_vars == 'WF':
        A2B_WF_abs = G_A2B(A, training=True)
        A2B_WF_abs = tf.where(A[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)
        A2B_PM = tf.zeros_like(B_PM)
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
    elif args.out_vars == 'PM':
        A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(B_PM!=0.0,A2B_PM,0.0)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_PM, A2B_PM)
    elif args.out_vars == 'WF-PM':
        A2B_abs = G_A2B(A, training=True)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_WF_abs,A2B_PM = tf.dynamic_partition(A2B_abs,indx_B,num_partitions=2)
            A2B_WF_abs = tf.reshape(A2B_WF_abs,B[:,:,:,:2].shape)
            A2B_PM = tf.reshape(A2B_PM,B[:,:,:,4:].shape)
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)
        val_sup_loss = sup_loss_fn(B_abs, A2B_abs)

    return A2B_abs, {'sup_loss': val_sup_loss}

def validation_step(A, B):
    A2B_abs, val_sup_dict = sample(A, B)
    return A2B_abs, val_sup_dict


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
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        p = np.random.rand()
        if p <= 0.4:
            # Random 90 deg rotations
            for _ in range(np.random.randint(3)):
                B = tf.image.rot90(B)

            # Random horizontal reflections
            B = tf.image.random_flip_left_right(B)

            # Random vertical reflections
            B = tf.image.random_flip_up_down(B)
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
            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A2B, val_sup_dict = validation_step(A, B)

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_sup_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
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
