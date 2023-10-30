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
py.arg('--n_G_filters', type=int, default=36)
py.arg('--n_downsamplings', type=int, default=4)
py.arg('--n_res_blocks', type=int, default=2)
py.arg('--encoded_size', type=int, default=256)
py.arg('--VQ_encoder', type=bool, default=False)
py.arg('--VQ_num_embed', type=int, default=64)
py.arg('--VQ_commit_cost', type=float, default=0.5)
py.arg('--adv_train', type=bool, default=False)
py.arg('--n_D_filters', type=int, default=72)
py.arg('--n_groups_D', type=int, default=1)
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
py.arg('--main_loss', default='MSE', choices=['MSE', 'MAE'])
py.arg('--A_loss', default='VGG', choices=['pix-wise', 'VGG', 'sinGAN'])
py.arg('--A_loss_weight', type=float, default=0.01)
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


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

A2Z2A_pool = data.ItemPool(args.pool_size)

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, 12, MEBCRN=True)

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, 12, MEBCRN=True)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, 12, MEBCRN=True)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, 12, MEBCRN=True)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, 12, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((acqs_1,acqs_3,acqs_4,acqs_5),axis=0)
valX    = acqs_2

# Overall dataset statistics
_,_,hgt,wdt,n_ch = np.shape(trainX)
trainX = trainX.reshape((-1,hgt,wdt,n_ch))
valX = valX.reshape((-1,hgt,wdt,n_ch))
len_dataset = trainX.shape[0]

print('Acquisition Dimensions:', hgt,wdt)

# Input dimensions (training & validation data)
print('Training input shape:',trainX.shape)
print('Validation input shape:',valX.shape)

A_dataset = tf.data.Dataset.from_tensor_slices(trainX)
A_dataset = A_dataset.batch(args.batch_size).shuffle(len_dataset)
A_dataset_val = tf.data.Dataset.from_tensor_slices(valX)
A_dataset_val.batch(1)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

enc= dl.encoder(input_shape=(hgt,wdt,n_ch),
                encoded_dims=args.encoded_size,
                multi_echo=False,
                filters=args.n_G_filters,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                sd_out=not(args.VQ_encoder),
                ls_reg_weight=args.ls_reg_weight,
                NL_self_attention=args.NL_SelfAttention
                )
dec= dl.decoder(encoded_dims=args.encoded_size,
                output_2D_shape=(hgt,wdt),
                multi_echo=False,
                filters=args.n_G_filters,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                output_activation=None,
                NL_self_attention=args.NL_SelfAttention
                )

D_A=dl.PatchGAN(input_shape=(hgt,wdt,2), 
                cGAN=False,
                multi_echo=False,
                dim=args.n_D_filters,
                self_attention=(args.NL_SelfAttention))

F_op = dl.FourierLayer(multi_echo=False)
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)
Cov_op = dl.CoVar()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')
if args.main_loss == 'MSE':
    cycle_loss_fn = tf.losses.MeanSquaredError()
elif args.main_loss == 'MAE':
    cycle_loss_fn = tf.losses.MeanAbsoluteError()
else:
    raise(NameError('Unrecognized Main Loss Function'))

cosine_loss = tf.losses.CosineSimilarity()
MSLE = tf.losses.MeanSquaredLogarithmicError()
if args.A_loss == 'VGG':
    metric_model = dl.perceptual_metric(input_shape=(hgt,wdt,n_ch), multi_echo=False)

if args.A_loss == 'sinGAN':
    # D_0 = dl.sGAN(input_shape=(None,None,n_ch))
    D_1 = dl.sGAN(input_shape=(None,None,n_ch))
    D_2 = dl.sGAN(input_shape=(None,None,n_ch))
    D_3 = dl.sGAN(input_shape=(None,None,n_ch))
    tl.Checkpoint(dict(D_1=D_1,D_2=D_2,D_3=D_3), py.join('output','sinGAN-WF','checkpoints')).restore()
    D_list = [D_1, D_2, D_3]
    batch_op = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))

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
def train_G(A):
    with tf.GradientTape(persistent=args.adv_train) as t:
        ##################### A Cycle #####################
        A2Z = enc(A, training=True)
        if args.VQ_encoder:
            vq_dict = vq_op(A2Z)
            A2Z = vq_dict['quantize']
        else:
            vq_dict =  {'loss': tf.constant(0.0,dtype=tf.float32),
                        'perplexity': tf.constant(0.0,dtype=tf.float32)}
        A2Z_cov = Cov_op(A2Z, training=False)
        A2Z2A = dec(A2Z, training=True)

        ############# Fourier Regularization ##############
        A_f = F_op(A, training=False)
        A2Z2A_f = F_op(A2Z2A, training=False)
        
        ############## Discriminative Losses ##############
        if args.adv_train:
            A2Z2A_d_logits = D_A(A2Z2A, training=False)
            A2Z2A_g_loss = g_loss_fn(A2Z2A_d_logits)
        else:
            A2Z2A_g_loss = tf.constant(0.0,dtype=tf.float32)
        
        ############ Cycle-Consistency Losses #############
        if args.A_loss == 'VGG':
            A2Y = metric_model(A, training=False)
            A2Z2A2Y = metric_model(A2Z2A, training=False)
            # A2Z2A_cycle_loss = cosine_loss(A2Y[0], A2Z2A2Y[0])/len(A2Y)
            A2Z2A_cycle_loss = cycle_loss_fn(A2Y[0], A2Z2A2Y[0])/len(A2Y)
            for l in range(1,len(A2Y)):
                # A2Z2A_cycle_loss += cosine_loss(A2Y[l], A2Z2A2Y[l])/len(A2Y)
                A2Z2A_cycle_loss += cycle_loss_fn(A2Y[l], A2Z2A2Y[l])/len(A2Y)
        elif args.A_loss == 'sinGAN':
            A2Z2A_cycle_loss = 0.0
            for D in D_list:
                A2Y = D(A, training=False)
                A2Z2A2Y = D(A2Z2A, training=False)
                for l in range(1,len(A2Y)):
                    A2Z2A_cycle_loss += cycle_loss_fn(A2Y[l], A2Z2A2Y[l])/(len(A2Y)*len(D_list))
        else:
            A2Z2A_cycle_loss = cycle_loss_fn(A, A2Z2A)
        A2Z2A_f_cycle_loss = cycle_loss_fn(A_f, A2Z2A_f)
        A2Z_cov_loss = cycle_loss_fn(A2Z_cov,tf.eye(A2Z_cov.shape[0]))

        ################ Regularizers #####################
        activ_reg = tf.constant(0.0,dtype=tf.float32)
        if enc.losses:
            activ_reg += tf.add_n(enc.losses)
        
        G_loss = args.A_loss_weight * A2Z2A_cycle_loss + A2Z2A_g_loss
        G_loss += activ_reg + A2Z2A_f_cycle_loss * args.Fourier_reg_weight + vq_dict['loss'] * args.ls_reg_weight
        G_loss += A2Z_cov_loss * args.cov_reg_weight
        
    G_grad = t.gradient(G_loss, enc.trainable_variables + dec.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, enc.trainable_variables + dec.trainable_variables))

    return A2Z2A,  {'A2Z2A_g_loss': A2Z2A_g_loss,
                    'A2Z2A_cycle_loss': A2Z2A_cycle_loss,
                    'A2Z2A_f_cycle_loss': A2Z2A_f_cycle_loss,
                    'LS_reg': activ_reg/args.ls_reg_weight,
                    'VQ_loss': vq_dict['loss'],
                    'VQ_perplexity': vq_dict['perplexity']}


@tf.function
def train_D(A, A2Z2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        A2Z2A_d_logits = D_A(A2Z2A, training=True)
        
        A_d_loss, A2Z2A_d_loss = d_loss_fn(A_d_logits, A2Z2A_d_logits)

        D_A_r1 = gan.R1_regularization(functools.partial(D_A, training=True), A)

        # D_Z_r2 = gan.R1_regularization(functools.partial(D_Z, training=True), A2Z.sample())

        D_loss = (A_d_loss + A2Z2A_d_loss) + (D_A_r1 * args.R1_reg_weight) #+ (D_Z_r2 * args.R2_reg_weight)

    D_grad = t.gradient(D_loss, D_A.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables))
    return {'D_loss': A_d_loss + A2Z2A_d_loss,
            'A_d_loss': A_d_loss,
            'A2Z2A_d_loss': A2Z2A_d_loss,
            'D_A_r1': D_A_r1,}
            # 'D_Z_r2': D_Z_r2}


def train_step(A):
    A2Z2A, G_loss_dict = train_G(A)

    if args.adv_train:
        # cannot autograph `A2Z2A_pool`
        A2Z2A = A2Z2A_pool(A2Z2A)
        for _ in range(args.critic_train_steps):
            D_loss_dict = train_D(A, A2Z2A)
    else:
        D_aux_val = tf.constant(0.0,dtype=tf.float32)
        D_loss_dict = {'D_loss': D_aux_val, 'A_d_loss': D_aux_val, 'A2Z2A_d_loss': D_aux_val}

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A):
    # A2Z2A Cycle
    A2Z = enc(A, training=False)
    if args.VQ_encoder:
        vq_dict = vq_op(A2Z)
        A2Z = vq_dict['quantize']
    A2Z2A = dec(A2Z, training=True)

    # Fourier regularization
    A_f = F_op(A, training=False)
    A2Z2A_f = F_op(A2Z2A, training=False)
    
    # Discriminative Losses
    if args.adv_train:
        A2Z2A_d_logits = D_A(A2Z2A, training=False)
        val_A2Z2A_g_loss = g_loss_fn(A2Z2A_d_logits)
    else:
        val_A2Z2A_g_loss = tf.constant(0.0,dtype=tf.float32)
    
    # Validation losses
    if args.A_loss == 'VGG':
        A2Y = metric_model(A, training=False)
        A2Z2A2Y = metric_model(A2Z2A, training=False)
        # val_A2Z2A_loss = cosine_loss(A2Y[0], A2Z2A2Y[0])/len(A2Y)
        val_A2Z2A_loss = cycle_loss_fn(A2Y[0], A2Z2A2Y[0])/len(A2Y)
        for l in range(1,len(A2Y)):
            # val_A2Z2A_loss += cosine_loss(A2Y[l], A2Z2A2Y[l])/len(A2Y)
            val_A2Z2A_loss += cycle_loss_fn(A2Y[l], A2Z2A2Y[l])/len(A2Y)
    elif args.A_loss == 'sinGAN':
        val_A2Z2A_loss = 0.0
        for D in D_list:
            A2Y = D(A, training=False)
            A2Z2A2Y = D(A2Z2A, training=False)
            for l in range(1,len(A2Y)):
                val_A2Z2A_loss += cycle_loss_fn(A2Y[l], A2Z2A2Y[l])/(len(A2Y)*len(D_list))
    else:
        val_A2Z2A_loss = cycle_loss_fn(A, A2Z2A)
    val_A2Z2A_f_loss = cycle_loss_fn(A_f, A2Z2A_f)
    return A2Z2A,  {'A2Z2A_g_loss': val_A2Z2A_g_loss,
                    'A2Z2A_cycle_loss': val_A2Z2A_loss,
                    'A2Z2A_f_cycle_loss': val_A2Z2A_f_loss}

def validation_step(A):
    A2Z2A, val_loss_dict = sample(A)
    return A2Z2A, val_loss_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(enc=enc,
                                dec=dec,
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
val_iter = cycle(A_dataset_val)
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
    for A in A_dataset:
        G_loss_dict, D_loss_dict = train_step(A)

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
            A = next(val_iter)
            A = tf.expand_dims(A,axis=0)
            A2Z2A, val_loss_dict = validation_step(A)

            # summary
            with val_summary_writer.as_default():
                tl.summary(val_loss_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(5,3), ncols=2)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(A[:,:,:,0],A[:,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(A2Z2A[:,:,:,0],A2Z2A[:,:,:,1])))
            
            # Acquisitions in the first row
            acq_ech1 = axs[0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0].set_title('Original')
            axs[0].axis('off')
            acq_ech2 = axs[1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[1].set_title('Recon')
            axs[1].axis('off')
            
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
