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
py.arg('--rand_ne', type=bool, default=False)
py.arg('--rand_ph_offset', type=bool, default=False)
py.arg('--unwrap', type=bool, default=True)
py.arg('--n_G_filters', type=int, default=36)
py.arg('--n_G_filt_list', default='')
py.arg('--n_downsamplings', type=int, default=4)
py.arg('--n_res_blocks', type=int, default=2)
py.arg('--encoded_size', type=int, default=256)
py.arg('--VQ_encoder', type=bool, default=False)
py.arg('--VQ_num_embed', type=int, default=64)
py.arg('--VQ_commit_cost', type=float, default=0.5)
py.arg('--adv_train', type=bool, default=False)
py.arg('--cGAN', type=bool, default=False)
py.arg('--n_D_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=20)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0002)
py.arg('--D_lr_factor', type=int, default=1)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--beta_2', type=float, default=0.9)
py.arg('--critic_train_steps', type=int, default=1)
py.arg('--R1_reg_weight', type=float, default=0.2)
py.arg('--main_loss', default='MSE', choices=['MSE', 'MAE', 'MSLE'])
py.arg('--A_loss', default='VGG', choices=['pix-wise', 'VGG'])
py.arg('--A_loss_weight', type=float, default=0.01)
py.arg('--B_loss_weight', type=float, default=0.1)
py.arg('--FM_loss_weight', type=float, default=1.0)
py.arg('--ls_reg_weight', type=float, default=1e-7)
py.arg('--cov_reg_weight', type=float, default=0.0)
py.arg('--Fourier_reg_weight', type=float, default=0.0)
py.arg('--NL_SelfAttention',type=bool, default=True)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

if len(args.n_G_filt_list) > 0:
    filt_list = [int(a_i) for a_i in args.n_G_filt_list.split(',')]

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A2B2A_pool = data.ItemPool(args.pool_size * (not args.rand_ne))

fm_sc = 300.0
r2_sc = 200.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'INTA_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, 12, MEBCRN=True,
                                    mag_and_phase=True, unwrap=args.unwrap)

dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, 12, MEBCRN=True,
                                    mag_and_phase=True, unwrap=args.unwrap)

dataset_hdf5_3 = 'Volunteers_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, 12, MEBCRN=True,
                                    mag_and_phase=True, unwrap=args.unwrap)

dataset_hdf5_4 = 'Attilio_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, 12, MEBCRN=True,
                                    mag_and_phase=True, unwrap=args.unwrap)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((acqs_2,acqs_3,acqs_4),axis=0)
valX    = acqs_1

trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4),axis=0)
valY    = out_maps_1

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(trainX)
_,_,_,_,n_out = np.shape(trainY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',ne)
print('Output Maps:',n_out)

# Input and output dimensions (training data)
print('Training input shape:',trainX.shape)
print('Training output shape:',trainY.shape)

# Input and output dimensions (validations data)
print('Validation input shape:',valX.shape)
print('Validation output shape:',valY.shape)

A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))
A_B_dataset = A_B_dataset.batch(args.batch_size).shuffle(len_dataset)
A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

# Num of decoders
nd = 3
if len(args.n_G_filt_list) == (args.n_downsamplings+1):
    nfe = filt_list
    nfd = [a//nd for a in filt_list]
    nfd2 = [a//(nd+1) for a in filt_list]
else:
    nfe = args.n_G_filters
    nfd = args.n_G_filters//nd
    nfd2= args.n_G_filters//(nd+1)
enc= dl.encoder(input_shape=(None,hgt,wdt,n_ch),
                encoded_dims=args.encoded_size,
                filters=nfe,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                sd_out=not(args.VQ_encoder),
                ls_mean_activ=None,
                ls_reg_weight=args.ls_reg_weight,
                NL_self_attention=args.NL_SelfAttention
                )
dec_ff  = dl.decoder(encoded_dims=args.encoded_size//3,
                    output_shape=(hgt,wdt,n_out-1),
                    filters=nfd2,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation='sigmoid',
                    output_initializer='he_normal',
                    NL_self_attention=args.NL_SelfAttention
                    )
dec_mag = dl.decoder(encoded_dims=args.encoded_size//3,
                    output_shape=(hgt,wdt,n_out),
                    filters=nfd,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation='relu',
                    output_initializer='he_normal',
                    NL_self_attention=args.NL_SelfAttention
                    )
dec_pha = dl.decoder(encoded_dims=args.encoded_size//3,
                    output_shape=(hgt,wdt,n_out),
                    filters=nfd,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation=None,
                    NL_self_attention=args.NL_SelfAttention
                    )

D_A=dl.PatchGAN(input_shape=(None,hgt,wdt,n_ch),
                cGAN=args.cGAN,
                multi_echo=True,
                dim=args.n_D_filters,
                self_attention=(args.NL_SelfAttention))

IDEAL_op = wf.IDEAL_mag_Layer()
APD_loss_fn = gan.AbsolutePhaseDisparity()

F_op = dl.FourierLayer()
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)
Cov_op = dl.CoVar()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')
if args.main_loss == 'MSE':
    cycle_loss_fn = tf.losses.MeanSquaredError()
elif args.main_loss == 'MAE':
    cycle_loss_fn = tf.losses.MeanAbsoluteError()
elif args.main_loss == 'MSLE':
    cycle_loss_fn = tf.losses.MeanSquaredLogarithmicError()
else:
    raise(NameError('Unrecognized Main Loss Function'))

cosine_loss = tf.losses.CosineSimilarity()
msle_loss = tf.losses.MeanSquaredLogarithmicError()
if args.A_loss == 'VGG':
    metric_model = dl.perceptual_metric(input_shape=(None,hgt,wdt,n_ch))

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
D_lr_scheduler = dl.LinearDecay(args.lr*args.D_lr_factor,
                                total_steps*args.critic_train_steps,
                                args.epoch_decay*total_steps*args.critic_train_steps / args.epochs)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape(persistent=args.adv_train) as t:
        ##################### A Cycle #####################
        A2Z = enc(A, training=True)
        A2Z_cov = Cov_op(A2Z, training=False)
        if args.VQ_encoder:
            vq_dict = vq_op(A2Z)
            A2Z = vq_dict['quantize']
        else:
            vq_dict =  {'loss': tf.constant(0.0,dtype=tf.float32),
                        'perplexity': tf.constant(0.0,dtype=tf.float32)}
        A2Z2B_ff = dec_ff(A2Z[...,:1], training=True)
        A2Z2B_mag = dec_mag(A2Z[...,1:2], training=True)
        A2Z2B_pha = dec_pha(A2Z[...,2:], training=True)
        
        A2Z2B_ff = tf.concat([A2Z2B_ff,tf.zeros_like(A2Z2B_ff)],axis=-1) # (NB,1,H,W,1+NS)
        A2B = tf.concat([A2Z2B_ff,A2Z2B_mag,A2Z2B_pha],axis=1)

        A2B2A = IDEAL_op(A2B, ne=A.shape[1], training=False)

        ############# Fourier Regularization ##############
        A_f = F_op(A, training=False)
        A2B2A_f = F_op(A2B2A, training=False)

        ############## Discriminative Losses ##############
        if args.adv_train:
            if args.cGAN:
                A_ref = A[:,0::2,:,:,:]
                A_g = A2B2A[:,1::2,:,:,:]
                if A_g.shape[1] < A_ref.shape[1]:
                    A_ref = A_ref[:,:-1,:,:,:]
                A2B2A_d_logits = D_A([A_g,A_ref], training=False)
            else:
                A2B2A_d_logits = D_A(A2B2A, training=False)
            A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)
        else:
            A2B2A_g_loss = tf.constant(0.0,dtype=tf.float32)

        ############ Cycle-Consistency Losses #############
        if args.A_loss == 'VGG':
            A2Y = metric_model(A, training=False)
            A2B2A2Y = metric_model(A2B2A, training=False)
            A2B2A_cycle_loss = tf.constant(0.0,dtype=tf.float32)
            for l in range(len(A2Y)):
                A2B2A_cycle_loss += cosine_loss(A2Y[l], A2B2A2Y[l])/len(A2Y)
        else:
            A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)

        B2A2B_cycle_loss = cycle_loss_fn(B[:,:2,...], A2B[:,:2,...]) # MAG
        B2A2B_cycle_loss += cycle_loss_fn(B[:,2:,...], A2B[:,2:,...]) * args.FM_loss_weight # PHASE
        
        A2B2A_f_cycle_loss = msle_loss(A_f, A2B2A_f)
        A2Z_cov_loss = cycle_loss_fn(A2Z_cov,tf.eye(A2Z_cov.shape[0]))

        ################ Regularizers #####################
        activ_reg = tf.constant(0.0,dtype=tf.float32)
        if enc.losses:
            activ_reg += tf.add_n(enc.losses)

        G_loss = args.A_loss_weight * A2B2A_cycle_loss + args.B_loss_weight * B2A2B_cycle_loss + A2B2A_g_loss
        G_loss += activ_reg + A2B2A_f_cycle_loss * args.Fourier_reg_weight + vq_dict['loss'] * args.ls_reg_weight
        G_loss += A2Z_cov_loss * args.cov_reg_weight

    G_grad = t.gradient(G_loss, enc.trainable_variables + dec_ff.trainable_variables + dec_mag.trainable_variables + dec_pha.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, enc.trainable_variables + dec_ff.trainable_variables + dec_mag.trainable_variables + dec_pha.trainable_variables))

    return A2B2A,  {'A2B2A_g_loss': A2B2A_g_loss,
                    'A2B2A_cycle_loss': A2B2A_cycle_loss,
                    'B2A2B_cycle_loss': B2A2B_cycle_loss,
                    'A2B2A_f_cycle_loss': A2B2A_f_cycle_loss,
                    'LS_reg': activ_reg/args.ls_reg_weight,
                    'Cov_reg': A2Z_cov_loss,
                    'VQ_loss': vq_dict['loss'],
                    'VQ_perplexity': vq_dict['perplexity']}


@tf.function
def train_D(A, A2B2A):
    with tf.GradientTape() as t:
        if args.cGAN:
            A_ref = A[:,0::2,:,:,:]
            A_r = A[:,1::2,:,:,:]
            A_f = A2B2A[:,1::2,:,:,:]
            if A_r.shape[1] < A_ref.shape[1]:
                A_ref = A_ref[:,:-1,:,:,:]
            A_d_logits = D_A([A_r,A_ref], training=True)
            A2B2A_d_logits = D_A([A_f,A_ref], training=True)
        else:
            A_d_logits = D_A(A, training=True)
            A2B2A_d_logits = D_A(A2B2A, training=True)

        A_d_loss, A2B2A_d_loss = d_loss_fn(A_d_logits, A2B2A_d_logits)

        if args.cGAN:
            D_A_r1 = gan.R1_regularization(functools.partial(D_A, training=True), [A_r,A_ref])
        else:
            D_A_r1 = gan.R1_regularization(functools.partial(D_A, training=True), A)

        # D_Z_r2 = gan.R1_regularization(functools.partial(D_Z, training=True), A2Z.sample())

        D_loss = (A_d_loss + A2B2A_d_loss) + (D_A_r1 * args.R1_reg_weight) #+ (D_Z_r2 * args.R2_reg_weight)

    D_grad = t.gradient(D_loss, D_A.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables))
    return {'D_loss': A_d_loss + A2B2A_d_loss,
            'A_d_loss': A_d_loss,
            'A2B2A_d_loss': A2B2A_d_loss,
            'D_A_r1': D_A_r1,}
            # 'D_Z_r2': D_Z_r2}


def train_step(A, B):
    A2B2A, G_loss_dict = train_G(A, B)

    if args.adv_train:
        # cannot autograph `A2B_pool`
        A2B2A = A2B2A_pool(A2B2A)
        for _ in range(args.critic_train_steps):
            D_loss_dict = train_D(A, A2B2A)
    else:
        D_aux_val = tf.constant(0.0,dtype=tf.float32)
        D_loss_dict = {'D_loss': D_aux_val, 'A_d_loss': D_aux_val, 'A2B2A_d_loss': D_aux_val}

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    # A2B2A Cycle
    A2Z = enc(A, training=False)
    if args.VQ_encoder:
        vq_dict = vq_op(A2Z)
        A2Z = vq_dict['quantize']
    A2Z2B_ff = dec_ff(A2Z[...,:1], training=False)
    A2Z2B_mag = dec_mag(A2Z[...,1:2], training=False)
    A2Z2B_pha = dec_pha(A2Z[...,2:], training=False)
    
    A2Z2B_ff = tf.concat([A2Z2B_ff,tf.zeros_like(A2Z2B_ff)],axis=-1) # (NB,1,H,W,NS)
    A2B = tf.concat([A2Z2B_ff,A2Z2B_mag,A2Z2B_pha],axis=1)
    
    A2B2A = IDEAL_op(A2B, training=False)

    # Fourier regularization
    A_f = F_op(A, training=False)
    A2B2A_f = F_op(A2B2A, training=False)

    # Discriminative Losses
    if args.adv_train:
        if args.cGAN:
            A_ref = A[:,0::2,:,:,:]
            A_g = A2B2A[:,1::2,:,:,:]
            A2B2A_d_logits = D_A([A_g,A_ref], training=False)
        else:
            A2B2A_d_logits = D_A(A2B2A, training=False)
        val_A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)
    else:
        val_A2B2A_g_loss = tf.constant(0.0,dtype=tf.float32)

    # Validation losses
    if args.A_loss == 'VGG':
        A2Y = metric_model(A, training=False)
        A2B2A2Y = metric_model(A2B2A, training=False)
        val_A2B2A_loss = tf.constant(0.0,dtype=tf.float32)
        for l in range(len(A2Y)):
            val_A2B2A_loss += cosine_loss(A2Y[l], A2B2A2Y[l])/len(A2Y)
    else:
        val_A2B2A_loss = cycle_loss_fn(A, A2B2A)
    
    val_B2A2B_loss = cycle_loss_fn(B[:,:2,...], A2B[:,:2,...])
    val_B2A2B_loss += cycle_loss_fn(B[:,2:,...], A2B[:,2:,...]) * args.FM_loss_weight
    
    val_A2B2A_f_loss = msle_loss(A_f, A2B2A_f)
    return A2B, A2B2A, {'A2B2A_g_loss': val_A2B2A_g_loss,
                        'A2B2A_cycle_loss': val_A2B2A_loss,
                        'B2A2B_cycle_loss': val_B2A2B_loss,
                        'A2B2A_f_cycle_loss': val_A2B2A_f_loss}

def validation_step(A, B):
    A2B, A2B2A, val_loss_dict = sample(A, B)
    return A2B, A2B2A, val_loss_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(enc=enc,
                                dec_mag=dec_mag,
                                dec_pha=dec_pha,
                                D_A=D_A,
                                vq_op=vq_op,
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
        if args.rand_ne:
            ne_sel = np.random.randint(3,7)
            A = A[:,:ne_sel,:,:,:]
        if args.rand_ph_offset:
            # TO BE UPDATED
            pha_offset = np.random.uniform(low=-np.pi/2,high=np.pi/2)
            A_mag = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
            A_pha = tf.math.atan2(A[...,1:],A[...,:1])
            A =  tf.concat([A_mag*tf.math.cos(A_pha+pha_offset),
                            A_mag*tf.math.sin(A_pha+pha_offset)],axis=-1)
            B_pha = B[:,1:,:,:,1:2] + pha_offset/np.pi
            if not unwrap:
                B_pha = tf.where(B_pha < -np.pi, B_pha + 2*np.pi, B_pha)
                B_pha = tf.where(B_pha > np.pi, B_pha - 2*np.pi, B_pha)
                B_out_pha = tf.concat([B_pha,B_pha,B[:,1:,:,:,2:]],axis=-1)
            B = tf.concat([B[:,:1,:,:,:],B_out_pha],axis=1)

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
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200//args.batch_size):
            A, B = next(val_iter)
            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A2B, A2B2A, val_loss_dict = validation_step(A, B)

            # summary
            with val_summary_writer.as_default():
                tl.summary(val_loss_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(A2B2A[:,0,:,:,0],A2B2A[:,0,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(A2B2A[:,1,:,:,0],A2B2A[:,1,:,:,1])))
            im_ech3 = np.squeeze(np.abs(tf.complex(A2B2A[:,2,:,:,0],A2B2A[:,2,:,:,1])))
            im_ech4 = np.squeeze(np.abs(tf.complex(A2B2A[:,3,:,:,0],A2B2A[:,3,:,:,1])))
            im_ech5 = np.squeeze(np.abs(tf.complex(A2B2A[:,4,:,:,0],A2B2A[:,4,:,:,1])))
            im_ech6 = np.squeeze(np.abs(tf.complex(A2B2A[:,5,:,:,0],A2B2A[:,5,:,:,1])))

            # Acquisitions in the first row
            acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,0].set_title('1st Echo')
            axs[0,0].axis('off')
            acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,1].set_title('2nd Echo')
            axs[0,1].axis('off')
            acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,2].set_title('3rd Echo')
            axs[0,2].axis('off')
            acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,3].set_title('4th Echo')
            axs[0,3].axis('off')
            acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,4].set_title('5th Echo')
            axs[0,4].axis('off')
            acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,5].set_title('6th Echo')
            axs[0,5].axis('off')

            # A2B maps in the second row
            w_m_aux = np.squeeze((1.0-A2B[:,0,:,:,0])*A2B[:,1,:,:,0])
            w_p_aux = np.squeeze(A2B[:,2,:,:,0])
            f_m_aux = np.squeeze(A2B[:,0,:,:,0]*A2B[:,1,:,:,0])
            f_p_aux = np.squeeze(A2B[:,2,:,:,0])
            r2_aux = np.squeeze(A2B[:,1,:,:,1])
            field_aux = np.squeeze(A2B[:,2,:,:,1])
            
            W_ok =  axs[1,0].imshow(w_m_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,0])
            axs[1,0].axis('off')

            Wp_ok =  axs[1,1].imshow(w_p_aux, cmap='twilight',
                                    interpolation='none', vmin=-1, vmax=1)
            fig.colorbar(Wp_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            F_ok =  axs[1,2].imshow(f_m_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            Fp_ok =  axs[1,3].imshow(f_p_aux, cmap='twilight',
                                    interpolation='none', vmin=-1, vmax=1)
            fig.colorbar(Fp_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            r2_ok = axs[1,4].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,4])
            axs[1,4].axis('off')

            field_ok =  axs[1,5].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,5])
            axs[1,5].axis('off')

            # Ground-truth in the third row
            wn_m_aux = np.squeeze((1-B[:,0,:,:,0])*B[:,1,:,:,0])
            wn_p_aux = np.squeeze(B[:,2,:,:,0])
            fn_m_aux = np.squeeze(B[:,0,:,:,0]*B[:,1,:,:,0])
            fn_p_aux = np.squeeze(B[:,2,:,:,0])
            r2n_aux = np.squeeze(B[:,1,:,:,1])
            fieldn_aux = np.squeeze(B[:,2,:,:,1])
            
            W_unet = axs[2,0].imshow(wn_m_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,0])
            axs[2,0].axis('off')

            Wp_unet = axs[2,1].imshow(wn_p_aux, cmap='twilight',
                                interpolation='none', vmin=-1, vmax=1)
            fig.colorbar(Wp_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            F_unet = axs[2,2].imshow(fn_m_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            Fp_unet = axs[2,3].imshow(fn_p_aux, cmap='twilight',
                                interpolation='none', vmin=-1, vmax=1)
            fig.colorbar(Fp_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            r2_unet = axs[2,4].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[2,4])
            axs[2,4].axis('off')

            field_unet = axs[2,5].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_unet, ax=axs[2,5])
            axs[2,5].axis('off')

            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
