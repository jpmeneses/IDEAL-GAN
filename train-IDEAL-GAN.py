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
py.arg('--adv_train', type=bool, default=False)
py.arg('--n_echoes', type=int, default=6)
py.arg('--G_model', default='encod-decod', choices=['multi-decod','encod-decod','U-Net','MEBCRN'])
py.arg('--n_G_filters', type=int, default=36)
py.arg('--n_downsamplings', type=int, default=4)
py.arg('--n_res_blocks', type=int, default=2)
py.arg('--n_D_filters', type=int, default=64)
py.arg('--encoded_size', type=int, default=256)
py.arg('--frac_labels', type=bool, default=False)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0002)
py.arg('--D_lr_factor', type=int, default=1)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--beta_2', type=float, default=0.9)
py.arg('--critic_train_steps', type=int, default=1)
py.arg('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--R1_reg_weight', type=float, default=0.2)
py.arg('--R2_reg_weight', type=float, default=0.2)
py.arg('--perceptual_loss', type=bool, default=True)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--B2A2B_weight', type=float, default=1.0)
py.arg('--ls_reg_weight', type=float, default=1.0)
py.arg('--Fourier_reg_weight', type=float, default=1e-5)
py.arg('--NL_SelfAttention',type=bool, default=False)
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

A2B2A_pool = data.ItemPool(args.pool_size)

ech_idx = args.n_echoes * 2
fm_sc = 300.0
r2_sc = 2*np.pi*fm_sc

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, ech_idx, MEBCRN=True)

dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx, MEBCRN=True)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx, MEBCRN=True)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx, MEBCRN=True)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((acqs_1,acqs_3,acqs_4,acqs_5),axis=0)
valX    = acqs_2

if args.frac_labels:
    n1_div,n3_div,n4_div = 384,730,888
else:
    n1_div,n3_div,n4_div = 0,0,0
trainY  = np.concatenate((out_maps_1[n1_div:,:,:,:],out_maps_3[n3_div:,:,:,:],out_maps_4[n4_div:,:,:,:],out_maps_5),axis=0)
valY    = out_maps_2

# Overall dataset statistics
len_dataset,_,hgt,wdt,n_ch = np.shape(trainX)
_,n_out,_,_,_ = np.shape(trainY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',args.n_echoes)
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

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(args.n_echoes,hgt,wdt,n_ch),
                    encoded_dims=args.encoded_size,
                    filters=args.n_G_filters,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    ls_reg_weight=args.ls_reg_weight,
                    NL_self_attention=args.NL_SelfAttention
                    )
    dec_w =  dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_f =  dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_xi = dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        NL_self_attention=args.NL_SelfAttention
                        )
else:
    raise(NameError('Unrecognized Generator Architecture'))

D_A=dl.PatchGAN(input_shape=(args.n_echoes,hgt,wdt,2), 
                multi_echo=True,
                dim=args.n_D_filters,
                self_attention=(args.NL_SelfAttention))

metric_model = dl.metric_model(input_shape=(args.n_echoes,hgt,wdt,n_ch))

IDEAL_op = wf.IDEAL_Layer(args.n_echoes,MEBCRN=True)
LWF_op = wf.LWF_Layer(args.n_echoes,MEBCRN=True)
F_op = dl.FourierLayer()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanSquaredError()
cosine_loss = tf.losses.CosineSimilarity()

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
    tf.debugging.check_numerics(A, message='Training input A numerical error')
    with tf.GradientTape(persistent=args.adv_train) as t:
        ##################### A Cycle #####################
        A2Z = enc(A, training=True)
        A2Z_var = tf.math.reduce_variance(A2Z)
        A2Z = tf.math.divide_no_nan(A2Z,A2Z_var)
        A2Z2B_w = dec_w(A2Z, training=True)
        A2Z2B_f = dec_f(A2Z, training=True)
        A2Z2B_xi= dec_xi(A2Z, training=True)
        A2B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)
        A2B2A = IDEAL_op(A2B, training=False)

        if args.adv_train:
            A2B_L = tf.concat([A2Z2B_w,A2Z2B_f],axis=1)
            A2B2A_L = LWF_op(A2B_L, training=False)
            tf.debugging.check_numerics(A2B2A_L, message='Linear A2B2A numerical error')
        else:
            A2B2A_L = A2B2A

        ##################### B Cycle #####################
        # B2A = IDEAL_op(B, training=False)
        # B2A2B = G_A2B(B2A, training=True)

        ############# Fourier Regularization ##############
        A_f = F_op(A, training=False)
        A2B2A_f = F_op(A2B2A, training=False)
        
        ############## Discriminative Losses ##############
        if args.adv_train:
            A2B2A_d_logits = D_A(A2B2A_L, training=False)
            tf.debugging.check_numerics(A2B2A_d_logits, message='Train_G: A2B2A D-logits numerical error')
            A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)
        else:
            A2B2A_g_loss = tf.constant(0.0,dtype=tf.float32)
        
        ############ Cycle-Consistency Losses #############
        if args.perceptual_loss:
            A2Y = metric_model(A, training=False)
            A2B2A2Y = metric_model(A2B2A, training=False)
            A2B2A_cycle_loss = cosine_loss(A2Y[0], A2B2A2Y[0])/len(A2Y)
            for l in range(1,len(A2Y)):
                A2B2A_cycle_loss += cosine_loss(A2Y[l], A2B2A2Y[l])/len(A2Y)
        else:
            A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, A2B)
        A2B2A_f_cycle_loss = cycle_loss_fn(A_f, A2B2A_f)

        ################ Regularizers #####################
        activ_reg = tf.add_n(enc.losses)
        
        G_loss = (A2B2A_cycle_loss + args.B2A2B_weight*B2A2B_cycle_loss)*args.cycle_loss_weight + A2B2A_g_loss
        G_loss += activ_reg
        G_loss += A2B2A_f_cycle_loss * args.Fourier_reg_weight
        
    G_grad = t.gradient(G_loss, enc.trainable_variables + dec_w.trainable_variables + dec_f.trainable_variables + dec_xi.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, enc.trainable_variables + dec_w.trainable_variables + dec_f.trainable_variables + dec_xi.trainable_variables))

    return A2B2A_L,{'A2B2A_g_loss': A2B2A_g_loss,
                    'A2B2A_cycle_loss': A2B2A_cycle_loss,
                    'B2A2B_cycle_loss': B2A2B_cycle_loss,
                    'A2B2A_f_cycle_loss': A2B2A_f_cycle_loss,
                    'LS_reg': activ_reg/args.ls_reg_weight}


@tf.function
def train_D(A, A2B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        tf.debugging.check_numerics(A_d_logits, message='A D-logits numerical error')
        A2B2A_d_logits = D_A(A2B2A, training=True)
        tf.debugging.check_numerics(A2B2A_d_logits, message='A2B2A D-logits numerical error')
        
        A_d_loss, A2B2A_d_loss = d_loss_fn(A_d_logits, A2B2A_d_logits)
        tf.debugging.check_numerics(A2B2A_d_loss, message='A2B2A D-loss numerical error')
        
        # D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, A2B2A, mode=args.gradient_penalty_mode)

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
            tf.debugging.check_numerics(A, message='Before D-step: A numerical error')
            tf.debugging.check_numerics(A2B2A, message='Before D-step: A2B2A numerical error')
            D_loss_dict = train_D(A, A2B2A)
    else:
        D_aux_val = tf.constant(0.0,dtype=tf.float32)
        D_loss_dict = {'D_loss': D_aux_val, 'A_d_loss': D_aux_val, 'A2B2A_d_loss': D_aux_val}

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    # A2B2A Cycle
    A2Z = enc(A, training=False)
    A2Z_var = tf.math.reduce_variance(A2Z)
    A2Z = tf.math.divide_no_nan(A2Z,A2Z_var)
    A2Z2B_w = dec_w(A2Z, training=True)
    A2Z2B_f = dec_f(A2Z, training=True)
    A2Z2B_xi= dec_xi(A2Z, training=True)
    A2B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)
    A2B2A = IDEAL_op(A2B, training=False)
    A2B_L = tf.concat([A2Z2B_w,A2Z2B_f],axis=1)
    A2B2A_L = LWF_op(A2B_L, training=False)

    # B2A2B Cycle
    # B2A = IDEAL_op(B, training=False)
    # B2A2B = G_A2B(B2A, training=False)

    # Fourier regularization
    A_f = F_op(A, training=False)
    A2B2A_f = F_op(A2B2A, training=False)
    
    # Discriminative Losses
    if args.adv_train:
        A2B2A_d_logits = D_A(A2B2A, training=False)
        val_A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)
    else:
        val_A2B2A_g_loss = tf.constant(0.0,dtype=tf.float32)
    
    # Validation losses
    if args.perceptual_loss:
        A2Y = metric_model(A, training=False)
        A2B2A2Y = metric_model(A2B2A, training=False)
        val_A2B2A_loss = cosine_loss(A2Y[0], A2B2A2Y[0])/len(A2Y)
        for l in range(1,len(A2Y)):
            val_A2B2A_loss += cosine_loss(A2Y[l], A2B2A2Y[l])/len(A2Y)
    else:
        val_A2B2A_loss = cycle_loss_fn(A, A2B2A)
    val_B2A2B_loss = cycle_loss_fn(B, A2B)
    val_A2B2A_f_loss = cycle_loss_fn(A_f, A2B2A_f)
    return A2B,A2B2A_L,{'A2B2A_g_loss': val_A2B2A_g_loss,
                        'A2B2A_cycle_loss': val_A2B2A_loss,
                        'B2A2B_cycle_loss': val_B2A2B_loss,
                        'A2B2A_f_cycle_loss': val_A2B2A_f_loss,}

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
                                dec_w=dec_w,
                                dec_f=dec_f,
                                dec_xi=dec_xi,
                                D_A=D_A,
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
        tf.debugging.check_numerics(A, message='Dataset-extracted A numerical error')
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
            A2B, A2B2A, val_loss_dict = validation_step(A, B)

            # summary
            with val_summary_writer.as_default():
                tl.summary(val_loss_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(A2B2A[:,0,:,:,0],A2B2A[:,0,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(A2B2A[:,1,:,:,0],A2B2A[:,1,:,:,1])))
            if args.n_echoes >= 3:
                im_ech3 = np.squeeze(np.abs(tf.complex(A2B2A[:,2,:,:,0],A2B2A[:,2,:,:,1])))
            if args.n_echoes >= 4:
                im_ech4 = np.squeeze(np.abs(tf.complex(A2B2A[:,3,:,:,0],A2B2A[:,3,:,:,1])))
            if args.n_echoes >= 5:
                im_ech5 = np.squeeze(np.abs(tf.complex(A2B2A[:,4,:,:,0],A2B2A[:,4,:,:,1])))
            if args.n_echoes >= 6:
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
            w_aux = np.squeeze(np.abs(tf.complex(A2B[:,0,:,:,0],A2B[:,0,:,:,1])))
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(np.abs(tf.complex(A2B[:,1,:,:,0],A2B[:,1,:,:,1])))
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(A2B[:,2,:,:,1])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=fm_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(A2B[:,2,:,:,0])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,4])
            axs[1,4].axis('off')
            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            wn_aux = np.squeeze(np.abs(tf.complex(B[:,0,:,:,0],B[:,0,:,:,1])))
            W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            fn_aux = np.squeeze(np.abs(tf.complex(B[:,1,:,:,0],B[:,1,:,:,1])))
            F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            r2n_aux = np.squeeze(B[:,2,:,:,1])
            r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=fm_sc)
            fig.colorbar(r2_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            fieldn_aux = np.squeeze(B[:,2,:,:,0])
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
