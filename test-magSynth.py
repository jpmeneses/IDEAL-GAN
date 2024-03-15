import functools
import itertools

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tf2lib as tl
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--te_input', type=bool, default=False)
py.arg('--LDM', type=bool, default=False)
py.arg('--infer_steps', type=int, default=10)
py.arg('--n_samples', type=int, default=60)
py.arg('--val_batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

if not(hasattr(args,'VQ_num_embed')):
    py.arg('--VQ_num_embed', type=bool, default=256)
    py.arg('--VQ_commit_cost', type=int, default=0.5)
    VQ_args = py.args()
    args.__dict__.update(VQ_args.__dict__)

if not(hasattr(args,'data_size')):
    py.arg('--data_size', type=int, default=192, choices=[192,384])
    ds_args = py.args()
    args.__dict__.update(ds_args.__dict__)

if not(hasattr(args,'div_decod')):
    py.arg('--div_decod', type=bool, default=False)
    dec_args = py.args()
    args.__dict__.update(dec_args.__dict__)

if hasattr(args,'n_G_filt_list'):
    if len(args.n_G_filt_list) > 0:
        filt_list = [int(a_i) for a_i in args.n_G_filt_list.split(',')]
    else:
        filt_list = list()

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

fm_sc = 300.0
r2_sc = 200.0

dataset_dir = '../datasets/'
dataset_hdf5_2 = 'JGalgani_GC_' + str(args.data_size) + '_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_2, 12, end=args.n_samples,
                            MEBCRN=True, mag_and_phase=args.only_mag)

len_dataset,ne,hgt,wdt,n_ch = valX.shape
if args.only_mag:
    _,_,_,_,n_out = np.shape(valY)
else:
    _,n_out,_,_,_ = np.shape(valY)

A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val = A_B_dataset_val.batch(args.val_batch_size)

z_std = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

if args.div_decod:
    if args.only_mag:
        nd = 2
    else:
        nd = 3
else:
    nd = 1
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
                ls_mean_activ=args.ls_mean_activ,
                NL_self_attention=args.NL_SelfAttention)
if args.only_mag:
    dec_mag = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_out),
                        filters=nfd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='relu',
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_pha = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_out-1),
                        filters=nfd2,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='tanh',
                        NL_self_attention=args.NL_SelfAttention
                        )
else:
    dec_w =  dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        filters=nfd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_f =  dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        filters=nfd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_xi = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        n_groups=args.n_groups_PM,
                        filters=nfd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )
if args.LDM:
    unet = dl.denoise_Unet(dim=args.n_ldm_filters, dim_mults=(1,2,4), channels=args.encoded_size)
    # Initiate unet
    if args.only_mag:
        hgt_ls = dec_mag.input_shape[1]
        wdt_ls = dec_mag.input_shape[2]
    else:
        hgt_ls = dec_w.input_shape[1]
        wdt_ls = dec_w.input_shape[2]
    test_images = tf.ones((1, hgt_ls, wdt_ls, args.encoded_size), dtype=tf.float32)
    test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
    k = unet(test_images, test_timestamps)
    # Checkpoint
    tl.Checkpoint(dict(unet=unet,z_std=z_std), py.join(args.experiment_dir, 'checkpoints_ldm')).restore()
    # create a fixed beta schedule
    if args.scheduler == 'linear':
        beta = np.linspace(args.beta_start, args.beta_end, args.n_timesteps)
        # this will be used as discussed in the reparameterization trick
        alpha = 1 - beta
        alpha_bar = np.cumprod(alpha, 0)
        alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
    elif args.scheduler == 'cosine':
        x = np.linspace(0, args.n_timesteps, args.n_timesteps + 1)
        alpha_bar = np.cos(((x / args.n_timesteps) + args.s_value) / (1 + args.s_value) * np.pi * 0.5) ** 2
        alpha_bar /= alpha_bar[0]
        alpha = np.clip(alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
        beta = 1.0 - alpha

if args.only_mag:
    IDEAL_op = wf.IDEAL_mag_Layer()
else:
    IDEAL_op = wf.IDEAL_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)

if args.only_mag:
    tl.Checkpoint(dict(enc=enc, dec_mag=dec_mag, dec_pha=dec_pha, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()
else:
    tl.Checkpoint(dict(enc=enc, dec_w=dec_w, dec_f=dec_f, dec_xi=dec_xi, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()


# @tf.function
def sample(A,Z_std):
    # Turn complex-valued CSE-MR image into only-magnitude
    # A_mag = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
    # A_pha = 0.25 * np.pi
    # A_pha = tf.random.uniform(A_mag.shape,minval=-np.pi,maxval=np.pi,seed=0)
    # A = tf.concat([A_mag*tf.math.cos(A_pha),A_mag*tf.math.sin(A_pha)],axis=-1)
    A_real = A[:,:,:,:,:1]
    A = tf.concat([A_real,tf.zeros_like(A_real)],axis=-1)
    A2Z = enc(A, training=False)
    if args.VQ_encoder:
        vq_dict = vq_op(A2Z)
        A2Z = vq_dict['quantize']
    if args.LDM:
        A2Z = tf.math.divide_no_nan(A2Z,Z_std)
        # Forward diffusion
        rng, tsrng = np.random.randint(0, 100000, size=(2,))
        A2Z, noise = dm.forward_noise(rng, A2Z, args.infer_steps, alpha_bar)
        # Reverse diffusion
        for i in range(args.infer_steps-1):
            t = np.expand_dims(np.array(args.infer_steps-i-1, np.int32), 0)
            pred_noise = unet(A2Z, t)
            A2Z = dm.ddpm(A2Z, pred_noise, t, alpha, alpha_bar, beta)
        A2Z = tf.math.multiply_no_nan(A2Z,Z_std)
    # Reconstruct to synthesize missing phase
    if args.only_mag:
        A2Z2B_mag = dec_mag(A2Z, training=False)
        A2Z2B_pha = dec_pha(A2Z, training=False)
        A2Z2B_pha = tf.concat([tf.zeros_like(A2Z2B_pha[:,:,:,:,:1]),A2Z2B_pha],axis=-1)
        A2B = tf.concat([A2Z2B_mag,A2Z2B_pha],axis=1)
    else:
        A2Z2B_w = dec_w(A2Z, training=False)
        A2Z2B_f = dec_f(A2Z, training=False)
        A2Z2B_xi= dec_xi(A2Z, training=False)
        A2B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)
    # Reconstructed multi-echo images
    A2B2A = IDEAL_op(A2B)

    return A2B, A2B2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'phase_synth')
py.mkdir(save_dir)

ms_ssim_scores = []
ssim_scores = []

k = 0
for A, B in A_B_dataset_val:
    # Get only-magnitude latent space
    A2B, A2B2A = sample(A,z_std)

    # SSIM metrics for pairs of synthetic data within batch
    # idx_pairs = list(itertools.combinations(range(A.shape[0]), 2))
    # for idx_a, idx_b in idx_pairs:
    #     ms_ssim_scores.append(tf.image.ssim_multiscale(Z2B2A[idx_a]+1.0, Z2B2A[idx_b]+1.0, 2))
    #     ssim_scores.append(tf.image.ssim(Z2B2A[idx_a]+1.0, Z2B2A[idx_b]+1.0, 2))

    if args.only_mag:
        w_m_aux = np.squeeze(A2B[:,0,:,:,0])
        w_p_aux = np.squeeze(A2B[:,1,:,:,0])
        f_m_aux = np.squeeze(A2B[:,0,:,:,1])
        f_p_aux = np.squeeze(A2B[:,1,:,:,1])
        r2_aux = np.squeeze(A2B[:,0,:,:,2])
        field_aux = np.squeeze(A2B[:,1,:,:,2])
    else:
        w_m_aux = np.squeeze(np.abs(tf.complex(A2B[:,0,:,:,0],A2B[:,0,:,:,1])))
        w_p_aux = np.squeeze(np.arctan2(A2B[:,0,:,:,1],A2B[:,0,:,:,0]))/np.pi
        f_m_aux = np.squeeze(np.abs(tf.complex(A2B[:,1,:,:,0],A2B[:,1,:,:,1])))
        f_p_aux = np.squeeze(np.arctan2(A2B[:,1,:,:,1],A2B[:,1,:,:,0]))/np.pi
        r2_aux = np.squeeze(A2B[:,2,:,:,1])
        field_aux = np.squeeze(A2B[:,2,:,:,0])

    im_ech1 = np.squeeze(np.abs(tf.complex(A2B2A[:,0,:,:,0],A2B2A[:,0,:,:,1])))
    im_ech2 = np.squeeze(np.abs(tf.complex(A2B2A[:,1,:,:,0],A2B2A[:,1,:,:,1])))
    im_ech3 = np.squeeze(np.abs(tf.complex(A2B2A[:,2,:,:,0],A2B2A[:,2,:,:,1])))
    im_ech4 = np.squeeze(np.abs(tf.complex(A2B2A[:,3,:,:,0],A2B2A[:,3,:,:,1])))
    im_ech5 = np.squeeze(np.abs(tf.complex(A2B2A[:,4,:,:,0],A2B2A[:,4,:,:,1])))
    im_ech6 = np.squeeze(np.abs(tf.complex(A2B2A[:,5,:,:,0],A2B2A[:,5,:,:,1])))

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

    # Z2B maps in the second row
    W_ok =  axs[1,0].imshow(w_m_aux, cmap='bone',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok, ax=axs[1,0])
    axs[1,0].axis('off')

    Wp_ok = axs[1,1].imshow(w_p_aux, cmap='twilight',
                            interpolation='none', vmin=-1, vmax=1)
    fig.colorbar(Wp_ok, ax=axs[1,1])
    axs[1,1].axis('off')

    F_ok =  axs[1,2].imshow(f_m_aux, cmap='pink',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok, ax=axs[1,2])
    axs[1,2].axis('off')

    Fp_ok = axs[1,3].imshow(f_p_aux, cmap='twilight',
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
    if args.only_mag:
        wn_m_aux = np.squeeze(B[:,0,:,:,0])
        wn_p_aux = np.squeeze(B[:,1,:,:,0])
        fn_m_aux = np.squeeze(B[:,0,:,:,1])
        fn_p_aux = np.squeeze(B[:,1,:,:,1])
        r2n_aux = np.squeeze(B[:,0,:,:,2])
        fieldn_aux = np.squeeze(B[:,1,:,:,2])
    else:
        wn_m_aux = np.squeeze(np.abs(tf.complex(B[:,0,:,:,0],B[:,0,:,:,1])))
        wn_p_aux = np.squeeze(np.arctan2(B[:,0,:,:,1],B[:,0,:,:,0]))/np.pi
        fn_m_aux = np.squeeze(np.abs(tf.complex(B[:,1,:,:,0],B[:,1,:,:,1])))
        fn_p_aux = np.squeeze(np.arctan2(B[:,1,:,:,1],B[:,1,:,:,0]))/np.pi
        r2n_aux = np.squeeze(B[:,2,:,:,1])
        fieldn_aux = np.squeeze(B[:,2,:,:,0])

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
                            interpolation='none', vmin=-2, vmax=2)
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

    # fig.suptitle('TE1/dTE: '+str([TE[0,0,0].numpy(),np.mean(np.diff(TE, axis=1))]), fontsize=16)

    # plt.show()
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.8)
    plt.savefig(save_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)

    k+=1

# ms_ssim_scores = tf.concat(ms_ssim_scores,axis=0)
# print(f"MS-SSIM Score: {tf.reduce_mean(ms_ssim_scores).numpy():.4f} +- {tf.math.reduce_std(ms_ssim_scores).numpy():.4f}")

# ssim_scores = tf.concat(ssim_scores,axis=0)
# print(f"SSIM Score: {tf.reduce_mean(ssim_scores).numpy():.4f} +- {tf.math.reduce_std(ssim_scores).numpy():.4f}")