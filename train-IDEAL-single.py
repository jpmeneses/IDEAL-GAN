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
py.arg('--grad_mode', default='bipolar', choices=['unipolar','bipolar'])
py.arg('--n_G_filters', type=int, default=36)
py.arg('--epochs', type=int, default=7000)
py.arg('--epoch_decay', type=int, default=7000)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=500)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0008)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--main_loss', default='MSE', choices=['MSE', 'MAE', 'MSLE'])
py.arg('--FM_TV_weight', type=float, default=0.0001)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=False)
py.arg('--D2_SelfAttention',type=bool, default=True)
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
if args.grad_mode == 'bipolar':
    dataset_hdf5_1 = 'Bip_NRef_384_complex_2D.hdf5'
    start, end = 3, 4
else:
    dataset_hdf5_1 = 'multiTE_GC_384_complex_2D.hdf5'
    start, end = 10, 15
X, Y, te=data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx=24,
                        start=start, end=end, te_data=True, MEBCRN=True,
                        mag_and_phase=True, unwrap=True)

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(X)
_,_,_,_,n_out = np.shape(Y)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',ne)
print('Output Maps:',n_out)

# Input and output dimensions (training data)
print('Training input shape:',X.shape)
print('Training output shape:',Y.shape)

A_B_dataset = tf.data.Dataset.from_tensor_slices((X,Y,te)).batch(len_dataset)
train_iter = cycle(A_B_dataset)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_mag = dl.UNet(input_shape=(ne,hgt,wdt,1),
                n_out=n_out,
                ME_layer=True,
                filters=args.n_G_filters,
                output_activation='sigmoid',
                self_attention=args.D1_SelfAttention)

G_pha = dl.UNet(input_shape=(ne,hgt,wdt,1),
                n_out=n_out,
                ME_layer=True,
                filters=args.n_G_filters,
                output_activation='linear',
                self_attention=args.D2_SelfAttention)

IDEAL_op = wf.IDEAL_mag_Layer()
APD_loss_fn = gan.AbsolutePhaseDisparity()

if args.main_loss == 'MSE':
    loss_fn = tf.losses.MeanSquaredError()
elif args.main_loss == 'MAE':
    loss_fn = tf.losses.MeanAbsoluteError()
elif args.main_loss == 'MSLE':
    loss_fn = tf.losses.MeanSquaredLogarithmicError()
else:
    raise(NameError('Unrecognized Main Loss Function'))

G_lr_scheduler = dl.LinearDecay(args.lr, args.epochs, args.epoch_decay)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B, te=None):
    A_mag = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
    A_pha = tf.math.atan2(A[...,1:],A[...,:1])
    with tf.GradientTape() as t:
        # Compute model's output
        A2B_mag = G_mag(A_mag, training=True)
        A2B_pha = G_pha(A_pha, training=True)

        A2B = tf.concat([A2B_mag,A2B_pha],axis=1)
        A2B = tf.where(B!=0.0,A2B,0.0)

        A2B_WF_abs = A2B[:,:1,:,:,:2]
        A2B_PM = A2B[...,2:]

        A2B2A = IDEAL_op(A2B, training=False)

        G_loss = A2B2A_cycle_loss = loss_fn(A, A2B2A)

        ############### Splited losses ####################
        WF_abs_loss = loss_fn(B[:,:1,:,:,:2], A2B[:,:1,:,:,:2])
        R2_loss = loss_fn(B[:,:1,:,:,2:], A2B[:,:1,:,:,2:])
        FM_loss = loss_fn(B[:,1:,:,:,2:], A2B[:,1:,:,:,2:])

        ################ Regularizers #####################
        FM_TV = tf.reduce_sum(tf.image.total_variation(A2B[:,1,:,:,2:]))
        FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B[:,1:,:,:,2:]),axis=(1,2,3,4)))
        G_loss += FM_TV * args.FM_TV_weight + FM_L1 * args.FM_L1_weight

    G_grad = t.gradient(G_loss, G_mag.trainable_variables + G_pha.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_mag.trainable_variables + G_pha.trainable_variables))

    return A2B_WF_abs, A2B_PM, {'A2B2A_cycle_loss': A2B2A_cycle_loss,
                                'WF_loss': WF_abs_loss,
                                'R2_loss': R2_loss,
                                'FM_loss': FM_loss,
                                'TV_FM': FM_TV,
                                'L1_FM': FM_L1}


def train_step(A, B, te=None):
    A2B_WF_abs, A2B_PM, G_loss_dict = train_G(A, B, te)
    return A2B_WF_abs, A2B_PM, G_loss_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_mag=G_mag,
                                G_pha=G_pha,
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

sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

A, B, TE = next(train_iter)

# Magnitude of recon MR images at each echo
im_ech1 = np.squeeze(np.abs(tf.complex(A[:1,0,:,:,0],A[:1,0,:,:,1])))
im_ech2 = np.squeeze(np.abs(tf.complex(A[:1,1,:,:,0],A[:1,1,:,:,1])))
if A.shape[1] >= 3:
    im_ech3 = np.squeeze(np.abs(tf.complex(A[:1,2,:,:,0],A[:1,2,:,:,1])))
if A.shape[1] >= 4:
    im_ech4 = np.squeeze(np.abs(tf.complex(A[:1,3,:,:,0],A[:1,3,:,:,1])))
if A.shape[1] >= 5:
    im_ech5 = np.squeeze(np.abs(tf.complex(A[:1,4,:,:,0],A[:1,4,:,:,1])))
if A.shape[1] >= 6:
    im_ech6 = np.squeeze(np.abs(tf.complex(A[:1,5,:,:,0],A[:1,5,:,:,1])))

# Ground-truth arrays
wn_aux = np.squeeze(B[:1,:1,:,:,0])
fn_aux = np.squeeze(B[:1,:1,:,:,1])
r2n_aux = np.squeeze(B[:1,:1,:,:,2])
fieldn_aux = np.squeeze(B[:1,1:,:,:,2])

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    A2B_WF_abs, A2B_PM, G_loss_dict = train_step(A, B, te=TE)

    # # summary
    with train_summary_writer.as_default():
        tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
        tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                    step=G_optimizer.iterations, name='G learning rate')

    # sample
    if (G_optimizer.iterations.numpy() % 1000 == 0) or (G_optimizer.iterations.numpy() < 50):
        fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

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
        w_aux = np.squeeze(A2B_WF_abs[:1,0,:,:,0])
        W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(W_ok, ax=axs[1,1])
        axs[1,1].axis('off')

        f_aux = np.squeeze(A2B_WF_abs[:1,0,:,:,1])
        F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(F_ok, ax=axs[1,2])
        axs[1,2].axis('off')

        r2_aux = np.squeeze(A2B_PM[:1,0,:,:,0])
        r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2_ok, ax=axs[1,3])
        axs[1,3].axis('off')

        field_aux = np.squeeze(A2B_PM[:1,0,:,:,0])
        field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
        fig.colorbar(field_ok, ax=axs[1,4])
        axs[1,4].axis('off')
        fig.delaxes(axs[1,0])
        fig.delaxes(axs[1,5])

        # Ground-truth in the third row
        W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                            interpolation='none', vmin=0, vmax=1)
        fig.colorbar(W_unet, ax=axs[2,1])
        axs[2,1].axis('off')

        F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                            interpolation='none', vmin=0, vmax=1)
        fig.colorbar(F_unet, ax=axs[2,2])
        axs[2,2].axis('off')

        r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                             interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2_unet, ax=axs[2,3])
        axs[2,3].axis('off')

        field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
        fig.colorbar(field_unet, ax=axs[2,4])
        axs[2,4].axis('off')
        fig.delaxes(axs[2,0])
        fig.delaxes(axs[2,5])

        fig.suptitle('TE1/dTE: '+str([TE[0,0,0].numpy(),np.mean(np.diff(TE[:1,...],axis=1))]), fontsize=16)

        # plt.show()
        plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
        tl.make_space_above(axs,topmargin=0.8)
        plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                    bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
