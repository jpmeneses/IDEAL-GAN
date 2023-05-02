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

py.arg('--dataset', default='WF-IDEAL')
py.arg('--n_echoes', type=int, default=6)
py.arg('--G_model', default='encod-decod', choices=['multi-decod','encod-decod','U-Net','MEBCRN'])
py.arg('--n_G_filters', type=int, default=36)
py.arg('--n_D_filters', type=int, default=72)
py.arg('--frac_labels', type=bool, default=False)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--beta_2', type=float, default=0.9)
py.arg('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--R1_reg_weight', type=float, default=0.2)
py.arg('--R2_reg_weight', type=float, default=0.2)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--B2A2B_weight', type=float, default=1.0)
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--FM_TV_weight', type=float, default=0.0)
py.arg('--R2_L1_weight', type=float, default=0.0)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=False)
py.arg('--D2_SelfAttention',type=bool, default=True)
py.arg('--D3_SelfAttention',type=bool, default=True)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A2B_pool = data.ItemPool(args.pool_size)

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, ech_idx)

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

n1_div = 248
n3_div = 0
n4_div = 434

trainX  = np.concatenate((acqs_1[n1_div:,:,:,:],acqs_3,acqs_4[n4_div:,:,:,:],acqs_5),axis=0)
valX    = acqs_2
testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)

valY    = out_maps_2
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

if args.frac_labels:
    n1_div = 384
    n3_div = 730
    n4_div = 888
trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3[n3_div:,:,:,:],out_maps_4[n4_div:,:,:,:],out_maps_5),axis=0)

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(trainX)
_,_,_,n_out = np.shape(trainY)
if args.G_model == 'complex':
    echoes = d_ech
else:
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

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(hgt,wdt,d_ech),
                    filters=args.n_G_filters,
                    te_shape=(args.n_echoes,),
                    )
    dec= dl.decoder(input_shape=enc.output_shape[1:],
                    n_out=6,
                    te_shape=(args.n_echoes,),
                    self_attention=args.D1_SelfAttention)
    G_A2B = tf.keras.Sequential()
    G_A2B.add(enc)
    G_A2B.add(dec)
else:
    raise(NameError('Unrecognized Generator Architecture'))

D_B = dl.PatchGAN(input_shape=(hgt,wdt,6), dim=args.n_D_filters, self_attention=(args.D1_SelfAttention))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanSquaredError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
D_lr_scheduler = dl.LinearDecay(4*args.lr, 5 * total_steps, 5 * args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    indices =tf.concat([tf.zeros_like(B[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32),
                        2*tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
    
    with tf.GradientTape() as t:
        ##################### A Cycle #####################
        A2B = G_A2B(A, training=True)

        # Split A2B param maps
        A2B_WF,A2B_R2,A2B_FM = tf.dynamic_partition(A2B,indices,num_partitions=3)
        A2B_WF = tf.reshape(A2B_WF,B[:,:,:,:4].shape)
        A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
        A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
        
        # Correct R2 scaling
        A2B_R2 = 0.5*A2B_R2 + 0.5
        A2B = tf.concat([A2B_WF,A2B_R2,A2B_FM],axis=-1)
        
        # Mask
        A2B = tf.where(B!=0.0,A2B,0.0)
        
        # Reconstructed multi-echo images
        A2B2A = wf.IDEAL_model(A2B,echoes)

        ##################### B Cycle #####################
        B2A = wf.IDEAL_model(B,echoes)
        B2A2B = G_A2B(B2A, training=True)
        
        # Split A2B param maps
        B2A2B_WF,B2A2B_R2,B2A2B_FM = tf.dynamic_partition(B2A2B,indices,num_partitions=3)
        B2A2B_WF = tf.reshape(B2A2B_WF,B[:,:,:,:4].shape)
        B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
        B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
        # Correct R2s scaling
        B2A2B_R2 = 0.5*B2A2B_R2 + 0.5
        B2A2B = tf.concat([B2A2B_WF,B2A2B_R2,B2A2B_FM],axis=-1)
        
        # B2A2B Mask
        B2A2B = tf.where(B!=0.0,B2A2B,0.0)

        ############## Discriminative Losses ##############
        A2B_d_logits = D_B(A2B, training=True)
        A2B_g_loss = g_loss_fn(A2B_d_logits)
        
        ############ Cycle-Consistency Losses #############
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)

        ################ Regularizers #####################
        R2_TV = tf.reduce_sum(tf.image.total_variation(A2B_R2)) * args.R2_TV_weight
        FM_TV = tf.reduce_sum(tf.image.total_variation(A2B_FM)) * args.FM_TV_weight
        R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_R2),axis=(1,2,3))) * args.FM_L1_weight
        FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_FM),axis=(1,2,3))) * args.FM_L1_weight
        reg_term = R2_TV + FM_TV + R2_L1 + FM_L1
        
        G_loss = (A2B_g_loss) + (A2B2A_cycle_loss + args.B2A2B_weight*B2A2B_cycle_loss)*args.cycle_loss_weight + reg_term
        
    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'TV_R2': R2_TV,
                      'L1_R2': R2_L1,
                      'TV_FM': FM_TV,
                      'L1_FM': FM_L1}


@tf.function
def train_D(B, A2B):
    with tf.GradientTape() as t:
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)
        
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_B_r1 = gan.R1_regularization(functools.partial(D_B, training=True), B)

        D_B_r2 = gan.R1_regularization(functools.partial(D_B, training=True), A2B)

        D_loss = (B_d_loss + A2B_d_loss) + (D_B_gp) * args.gradient_penalty_weight + (D_B_r1) * args.R1_reg_weight + (D_B_r2) * args.R2_reg_weight

    D_grad = t.gradient(D_loss, D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_B.trainable_variables))
    return {'D_loss': B_d_loss + A2B_d_loss,
            'B_d_loss': B_d_loss,
            'A2B_d_loss': A2B_d_loss,
            'D_B_gp': D_B_gp,
            'D_B_r1': D_B_r1,
            'D_B_r2': D_B_r2,}


def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)

    for _ in range(5):
        D_loss_dict = train_D(B, A2B)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    indices =tf.concat([tf.zeros_like(B[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32),
                        2*tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)

    # A2B2A Cycle
    A2B = G_A2B(A, training=False)
    # Split A2B param maps
    A2B_WF,A2B_R2,A2B_FM = tf.dynamic_partition(A2B,indices,num_partitions=3)
    A2B_WF = tf.reshape(A2B_WF,B[:,:,:,:4].shape)
    A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
    A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
    # Correct R2 scaling
    A2B_R2 = 0.5*A2B_R2 + 0.5
    A2B = tf.concat([A2B_WF,A2B_R2,A2B_FM],axis=-1)
    # Mask
    A2B = tf.where(B!=0.0,A2B,0.0)
    # Reconstructed multi-echo images
    A2B2A = wf.IDEAL_model(A2B,echoes)

    # B2A2B Cycle
    B2A = wf.IDEAL_model(B,echoes)
    B2A2B = G_A2B(B2A, training=False)
    # Split B2A2B param maps
    B2A2B_WF,B2A2B_R2,B2A2B_FM = tf.dynamic_partition(B2A2B,indices,num_partitions=3)
    B2A2B_WF = tf.reshape(B2A2B_WF,B[:,:,:,:4].shape)
    B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
    B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
    # Correct R2 scaling
    B2A2B_R2 = 0.5*B2A2B_R2 + 0.5
    B2A2B = tf.concat([B2A2B_WF,B2A2B_R2,B2A2B_FM],axis=-1)
    
    # Validation losses
    val_A2B2A_loss = tf.abs(cycle_loss_fn(A, A2B2A))
    val_B2A2B_loss = cycle_loss_fn(B, B2A2B)
    return A2B, B2A, A2B2A, B2A2B, {'A2B2A_cycle_loss': val_A2B2A_loss,
                                    'B2A2B_cycle_loss': val_B2A2B_loss}

def validation_step(A, B):
    A2B, B2A, A2B2A, B2A2B, val_loss_dict = sample(A, B)
    return A2B, B2A, A2B2A, B2A2B, val_loss_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
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
                A = tf.image.rot90(A)
            for _ in range(np.random.randint(3)):
                B = tf.image.rot90(B)

            # Random horizontal reflections
            A = tf.image.random_flip_left_right(A)
            B = tf.image.random_flip_left_right(B)

            # Random vertical reflections
            A = tf.image.random_flip_up_down(A)
            B = tf.image.random_flip_up_down(B)
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        G_loss_dict, D_loss_dict = train_step(A, B)

        # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')
            tl.summary({'D learning rate': D_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='D learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200):
            A, B = next(val_iter)
            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A2B, B2A, A2B2A, B2A2B, val_loss_dict = validation_step(A, B)

            # summary
            with val_summary_writer.as_default():
                tl.summary(val_loss_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,0],B2A[:,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,2],B2A[:,:,:,3])))
            if args.n_echoes >= 3:
                im_ech3 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,4],B2A[:,:,:,5])))
            if args.n_echoes >= 4:
                im_ech4 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,6],B2A[:,:,:,7])))
            if args.n_echoes >= 5:
                im_ech5 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,8],B2A[:,:,:,9])))
            if args.n_echoes >= 6:
                im_ech6 = np.squeeze(np.abs(tf.complex(B2A[:,:,:,10],B2A[:,:,:,11])))
            
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
            w_aux = np.squeeze(np.abs(tf.complex(B2A2B[:,:,:,0],B2A2B[:,:,:,1])))
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(np.abs(tf.complex(B2A2B[:,:,:,2],B2A2B[:,:,:,3])))
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(B2A2B[:,:,:,4])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(B2A2B[:,:,:,5])
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
