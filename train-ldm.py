import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import data
import DLlib as dl
import DMlib as dm
import pylib as py
import tf2lib as tl
import wflib as wf

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='GAN-100')
py.arg('--n_timesteps', type=int, default=200)
py.arg('--n_ldm_filters', type=int, default=64)
py.arg('--batch_size', type=int, default=8)
py.arg('--epochs_ldm', type=int, default=100)
py.arg('--epoch_ldm_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--data_augmentation', type=bool, default=False)
py.arg('--lr', type=float, default=0.0001)
ldm_args = py.args()

output_dir = py.join('output',ldm_args.experiment_dir)
args = py.args_from_yaml(py.join('output', ldm_args.experiment_dir, 'settings.yml'))
args.__dict__.update(ldm_args.__dict__)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, MEBCRN=True)

# dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
# acqs_2, out_maps_2 = data.load_hdf5(dataset_dir, dataset_hdf5_2, end=10, MEBCRN=True)

dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir, dataset_hdf5_3, MEBCRN=True)

dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir, dataset_hdf5_4, MEBCRN=True)

dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'
acqs_5, out_maps_5 = data.load_hdf5(dataset_dir, dataset_hdf5_5, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((acqs_1,acqs_3,acqs_4,acqs_5),axis=0)
trainY  = np.concatenate((out_maps_1,out_maps_3,out_maps_4,out_maps_5),axis=0)

# Overall dataset statistics
len_dataset,_,hgt,wdt,n_ch = np.shape(trainX)
_,n_out,_,_,_ = np.shape(trainY)

print('Image Dimensions:', hgt, wdt)
print('Num. Output Maps:',n_out)

A_dataset = tf.data.Dataset.from_tensor_slices(trainX)
A_dataset = A_dataset.batch(args.batch_size).shuffle(len_dataset)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(args.n_echoes,hgt,wdt,n_ch),
                    encoded_dims=args.encoded_size,
                    filters=args.n_G_filters,
                    num_layers=args.n_downsamplings,
                    ls_reg_weight=args.ls_reg_weight,
                    NL_self_attention=args.NL_SelfAttention
                    )
    dec_w =  dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_f =  dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_xi = dl.decoder(encoded_dims=args.encoded_size,
                        output_2D_shape=(hgt,wdt),
                        filters=args.n_G_filters,
                        num_layers=args.n_downsamplings,
                        NL_self_attention=args.NL_SelfAttention
                        )
else:
    raise(NameError('Unrecognized Generator Architecture'))

# create our unet model
unet = dl.denoise_Unet(dim=args.n_ldm_filters, channels=args.encoded_size)

IDEAL_op = wf.IDEAL_Layer(args.n_echoes,MEBCRN=True)

tl.Checkpoint(dict(enc=enc, dec_w=dec_w, dec_f=dec_f, dec_xi=dec_xi), py.join(args.experiment_dir, 'checkpoints')).restore()

################################################################################
########################### DIFFUSION TIMESTEPS ################################
################################################################################

# create a fixed beta schedule
beta = np.linspace(0.0001, 0.02, args.n_timesteps)

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)

# initialize the model in the memory of our GPU
hgt_ls = dec_w.input_shape[1]
wdt_ls = dec_w.input_shape[2]
test_images = np.ones([1, hgt_ls, wdt_ls, args.encoded_size])
test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
k = unet(test_images, test_timestamps)

loss_fn = tf.losses.MeanSquaredError()

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

def train_step(A):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = dm.generate_timestamp(tsrng, A.shape[0], args.n_timesteps)

    with tf.GradientTape() as t:
        A2Z = enc(A, training=False)
        Z_n, noise = dm.forward_noise(rng, A2Z, timestep_values, alpha_bar)
        pred_noise = unet(Z_n, timestep_values)
        
        loss_value = loss_fn(noise, pred_noise)
    
    gradients = t.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return {'Loss': loss_value}

def validation_step(Z):
    for i in range(args.n_timesteps-1):
        t = np.expand_dims(np.array(args.n_timesteps-i-1, np.int32), 0)
        pred_noise = unet(Z, t)
        Z = dm.ddpm(Z, pred_noise, t, alpha, alpha_bar, beta)

    Z2B_w = dec_w(Z, training=False)
    Z2B_f = dec_f(Z, training=False)
    Z2B_xi= dec_xi(Z, training=False)
    Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
    Z2B2A = IDEAL_op(Z2B, training=False)

    return Z2B, Z2B2A

# epoch counter
ep_cnt_ldm = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint_ldm = tl.Checkpoint(dict(unet=unet,
                                    optimizer=opt,
                                    ep_cnt=ep_cnt_ldm),
                               py.join(output_dir, 'checkpoints_ldm'),
                               max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint_ldm.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'LDM'))

# sample
sample_dir = py.join(output_dir, 'samples_ldm_training')
py.mkdir(sample_dir)

r2_sc, fm_sc = 200.0, 300.0

# main loop
for ep in range(args.epochs_ldm):
    if ep < ep_cnt_ldm:
        continue

    # update epoch counter
    ep_cnt_ldm.assign_add(1)

    # train for an epoch
    for A in A_dataset:
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        for i in range(A.shape[0]):
            if args.data_augmentation:
                A_i = A[i,:,:,:,:]
                p = np.random.rand()
                if p <= 0.4:
                    # Random 90 deg rotations
                    A_i = tf.image.rot90(A_i,k=np.random.randint(3))

                    # Random horizontal reflections
                    A_i = tf.image.random_flip_left_right(A_i)

                    # Random vertical reflections
                    A_i = tf.image.random_flip_up_down(A_i)
                
                A_i = tf.expand_dims(A_i,axis=0)
                if i <= 0:
                    A_da = tf.expand_dims(A_i,axis=0)
                else:
                    A_da = tf.concat([A_da, tf.expand_dims(A_i, axis=0)],axis=0)
            else:
                A_da = A
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        loss_dict = train_step(A_da)

        # summary
        with train_summary_writer.as_default():
            tl.summary(loss_dict, step=opt.iterations, name='LDM_losses')

    if (((ep+1) % args.epoch_ldm_ckpt) == 0) or ((ep+1)==args.epochs_ldm):
        checkpoint_ldm.save(ep)

    # Validation inference
    Z = tf.random.normal((1,hgt_ls,wdt_ls,args.encoded_size), dtype=tf.float32)
    Z2B, Z2B2A = validation_step(Z)

    fig, axs = plt.subplots(figsize=(20, 6), nrows=2, ncols=6)

    # Magnitude of recon MR images at each echo
    im_ech1 = np.squeeze(np.abs(tf.complex(Z2B2A[0,:,:,0],Z2B2A[0,:,:,1])))
    im_ech2 = np.squeeze(np.abs(tf.complex(Z2B2A[1,:,:,0],Z2B2A[1,:,:,1])))
    if args.n_echoes >= 3:
        im_ech3 = np.squeeze(np.abs(tf.complex(Z2B2A[2,:,:,0],Z2B2A[2,:,:,1])))
    if args.n_echoes >= 4:
        im_ech4 = np.squeeze(np.abs(tf.complex(Z2B2A[3,:,:,0],Z2B2A[3,:,:,1])))
    if args.n_echoes >= 5:
        im_ech5 = np.squeeze(np.abs(tf.complex(Z2B2A[4,:,:,0],Z2B2A[4,:,:,1])))
    if args.n_echoes >= 6:
        im_ech6 = np.squeeze(np.abs(tf.complex(Z2B2A[5,:,:,0],Z2B2A[5,:,:,1])))
    
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
    w_aux = np.squeeze(np.abs(tf.complex(Z2B[:,0,:,:,0],Z2B[:,0,:,:,1])))
    W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok, ax=axs[1,1])
    axs[1,1].axis('off')

    f_aux = np.squeeze(np.abs(tf.complex(Z2B[:,1,:,:,0],Z2B[:,1,:,:,1])))
    F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok, ax=axs[1,2])
    axs[1,2].axis('off')

    r2_aux = np.squeeze(Z2B[:,2,:,:,1])
    r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                            interpolation='none', vmin=0, vmax=fm_sc)
    fig.colorbar(r2_ok, ax=axs[1,3])
    axs[1,3].axis('off')

    field_aux = np.squeeze(Z2B[:,2,:,:,0])
    field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_ok, ax=axs[1,4])
    axs[1,4].axis('off')
    fig.delaxes(axs[1,0])
    fig.delaxes(axs[1,5])

    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.8)
    plt.savefig(py.join(sample_dir, 'ep-%03d.png' % (ep+1)), bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

