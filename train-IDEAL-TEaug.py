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
import mebcrn
from keras_unet.models import custom_unet

from itertools import cycle

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='WF-IDEAL')
py.arg('--n_echoes', type=int, default=6)
py.arg('--G_model', default='encod-decod', choices=['encod-decod','complex'])
py.arg('--n_G_filters', type=int, default=72)
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
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--FM_TV_weight', type=float, default=0.0)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--R2_SelfAttention',type=bool, default=False)
py.arg('--FM_SelfAttention',type=bool, default=True)
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

A2B_R2_pool = data.ItemPool(args.pool_size)
A2B_FM_pool = data.ItemPool(args.pool_size)

if args.G_model == 'complex':
    ech_idx = args.n_echoes
else:
    ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx,
                            acqs_data=False, te_data=False,
                            complex_data=(args.G_model=='complex'))

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                            acqs_data=False, te_data=False,
                            complex_data=(args.G_model=='complex'))

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx,
                            acqs_data=False, te_data=False,
                            complex_data=(args.G_model=='complex'))

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                            acqs_data=False, te_data=False,
                            complex_data=(args.G_model=='complex'))

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx,
                            acqs_data=False, te_data=False,
                            complex_data=(args.G_model=='complex'))

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

n1_div = 248
n3_div = 0
n4_div = 434

trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3[n3_div:,:,:,:],out_maps_4[n4_div:,:,:,:]),axis=0)
valY    = out_maps_2
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Overall dataset statistics
len_dataset,hgt,wdt,n_out = np.shape(trainY)
echoes = args.n_echoes
if args.G_model == 'complex':
    d_ech = echoes
else:
    d_ech = echoes*2

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (training data)
print('Training output shape:',trainY.shape)

# Input and output dimensions (validations data)
print('Validation output shape:',valY.shape)

# Input and output dimensions (testing data)
print('Testing output shape:',testY.shape)

B_dataset = tf.data.Dataset.from_tensor_slices(trainY)
B_dataset = B_dataset.batch(args.batch_size).shuffle(len(trainY))
B_dataset_val = tf.data.Dataset.from_tensor_slices(valY)
B_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

if args.G_model == 'encod-decod':
    G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                            filters=args.n_G_filters,
                            te_input=True,
                            te_shape=(args.n_echoes,),
                            R2_self_attention=args.R2_SelfAttention,
                            FM_self_attention=args.FM_SelfAttention)
elif args.G_model == 'complex':
    G_A2B=dl.PM_complex(input_shape=(hgt,wdt,d_ech),
                        filters=args.n_G_filters,
                        te_input=True,
                        te_shape=(args.n_echoes,),
                        self_attention=(args.R2_SelfAttention and args.FM_SelfAttention))
else:
    raise(NameError('Unrecognized Generator Architecture'))

cycle_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(B, te=None):
    indx_B = tf.concat([tf.zeros_like(B[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:2],dtype=tf.int32)],axis=-1)

    indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)

    with tf.GradientTape() as t:
        ##################### B Cycle #####################
        # Split B outputs
        B_WF,B_PM = tf.dynamic_partition(B,indx_B,num_partitions=2)
        B_PM = tf.reshape(B_PM,B[:,:,:,4:].shape)

        B2A = wf.IDEAL_model(B,echoes,te=te)
        B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
        B2A2B_PM = G_A2B([B2A,te], training=True)
        # B2A2B_WF = wf.get_rho(B2A,B2A2B_PM)

        if args.G_model != 'complex':
            # B2A2B Mask
            B2A2B_PM = tf.where(B_PM!=0.0,B2A2B_PM,0.0)
            # Split A2B param maps
            B2A2B_R2,B2A2B_FM = tf.dynamic_partition(B2A2B_PM,indx_PM,num_partitions=2)
            B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
            B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
        else:
            B2A2B_R2 = tf.math.real(B2A2B_PM)
            B2A2B_FM = tf.math.imag(B2A2B_PM)
            B2A2B_PM = tf.concat([B2A2B_R2,B2A2B_FM],axis=-1)
            # B2A2B Mask
            B2A2B_PM = tf.where(B_PM!=0.0,B2A2B_PM,0.0)

        # Merge B2A2B output maps 
        # B2A2B = tf.concat([B2A2B_WF,B2A2B_PM],axis=-1)

        ############ Cycle-Consistency Losses #############
        B2A2B_cycle_loss = cycle_loss_fn(B_PM, B2A2B_PM)

        ################ Regularizers #####################
        R2_TV = tf.reduce_sum(tf.image.total_variation(B2A2B_R2)) * args.R2_TV_weight
        FM_TV = tf.reduce_sum(tf.image.total_variation(B2A2B_FM)) * args.FM_TV_weight
        FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(B2A2B_FM),axis=(1,2,3))) * args.FM_L1_weight # * (1-ep/(args.epochs-1))
        reg_term = R2_TV + FM_TV + FM_L1
        
        G_loss = args.B2A2B_weight*args.cycle_loss_weight + reg_term
        
    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return {'B2A2B_cycle_loss': B2A2B_cycle_loss,
            'TV_R2': R2_TV}


def train_step(B, te=None):
    G_loss_dict = train_G(B, te)
    return G_loss_dict


@tf.function
def sample(B, te=None):
    indx_B = tf.concat([tf.zeros_like(B[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:2],dtype=tf.int32)],axis=-1)
    indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
    B2A = wf.IDEAL_model(B,echoes,te=te_B)
    B2A = keras.layers.GaussianNoise(stddev=0.1)(B2A)
    B2A2B_PM = G_A2B([B2A,te], training=False)
    # B2A2B Mask
    B2A2B_PM = tf.where(B!=0.0,B2A2B_PM,0.0)
    B2A2B_WF = wf.get_rho(B2A,B2A2B_PM,te=te)
    B2A2B = tf.concat([B2A2B_WF,B2A2B_PM],axis=-1)

    B_WF,B_PM = tf.dynamic_partition(B,indx_B,num_partitions=2)
    B_PM = tf.reshape(B_PM,B[:,:,:,4:].shape)
    val_B2A2B_loss = cycle_loss_fn(B_PM, B2A2B_PM)
    return B2A, B2A2B, {'B2A2B_cycle_loss': val_B2A2B_loss}

def validation_step(B, TE):
    B2A, B2A2B, val_B2A2B_dict = sample(B, TE)
    return B2A, B2A2B, val_B2A2B_dict


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
val_iter = cycle(B_dataset_val)
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
            for _ in range(np.random.randint(3)):
                B = tf.image.rot90(B)

            # Random horizontal reflections
            B = tf.image.random_flip_left_right(B)

            # Random vertical reflections
            B = tf.image.random_flip_up_down(B)
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        te_var = wf.gen_TEvar(args.n_echoes,args.batch_size)

        G_loss_dict = train_step(B, te=te_var)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200):
            B = next(val_iter)
            B = tf.expand_dims(B,axis=0)
            TE_valid = wf.gen_TEvar(args.n_echoes,1)
            B2A, B2A2B, val_A2B_dict = validation_step(B, TE_valid)

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_A2B_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            if args.G_model != 'complex':
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
            else:
                im_ech1 = np.squeeze(np.abs(B2A[:,:,:,0]))
                im_ech2 = np.squeeze(np.abs(B2A[:,:,:,1]))
                if args.n_echoes >= 3:
                    im_ech3 = np.squeeze(np.abs(B2A[:,:,:,2]))
                if args.n_echoes >= 4:
                    im_ech4 = np.squeeze(np.abs(B2A[:,:,:,3]))
                if args.n_echoes >= 5:
                    im_ech5 = np.squeeze(np.abs(B2A[:,:,:,4]))
                if args.n_echoes >= 6:
                    im_ech6 = np.squeeze(np.abs(B2A[:,:,:,5]))
            
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
                                    interpolation='none')#, vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(np.abs(tf.complex(B2A2B[:,:,:,2],B2A2B[:,:,:,3])))
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none')#, vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(B2A2B[:,:,:,4])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none')#, vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(B2A2B[:,:,:,5])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none')#, vmin=-fm_sc/2, vmax=fm_sc/2)
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

            fig.suptitle('TE1/dTE: '+str([TE_valid[0,0].numpy(),np.mean(np.diff(TE_valid))]), fontsize=16)

            # plt.show()
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
