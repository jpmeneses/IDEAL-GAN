import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm
from skimage.restoration import unwrap_phase

import tf2lib as tl
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='output/GAN-100')
py.arg('--LDM', type=bool, default=True)
py.arg('--DDIM', type=bool, default=False)
py.arg('--infer_steps', type=int, default=10)
py.arg('--infer_sigma', type=float, default=0.0)
py.arg('--n_samples', type=int, default=50)
py.arg('--seed', type=int, default=0)
ldm_args = py.args()

output_dir = ldm_args.experiment_dir
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(ldm_args.__dict__)

if not(hasattr(args,'VQ_num_embed')):
    py.arg('--VQ_num_embed', type=bool, default=256)
    py.arg('--VQ_commit_cost', type=int, default=0.5)
    VQ_args = py.args()
    args.__dict__.update(VQ_args.__dict__)

if hasattr(args,'n_G_filt_list'):
    if len(args.n_G_filt_list) > 0:
        filt_list = [int(a_i) for a_i in args.n_G_filt_list.split(',')]
    else:
        filt_list = list()

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

fm_sc = 300.0
r2_sc = 200.0
hgt = args.data_size
wdt = args.data_size
n_ch = 2
n_out = 3

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

nd = 2
if len(args.n_G_filt_list) == (args.n_downsamplings+1):
    nfd = [a//nd for a in filt_list]
    nfd2 = [a//(nd+1) for a in filt_list]
else:
    nfd = args.n_G_filters//nd
    nfd2= args.n_G_filters//(nd+1)
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

# create our unet model
if args.LDM:
    unet = dl.denoise_Unet(dim=args.n_ldm_filters, dim_mults=(1,2,4), channels=args.encoded_size)

IDEAL_op = wf.IDEAL_mag_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size, args.VQ_num_embed, args.VQ_commit_cost)

tl.Checkpoint(dict(dec_mag=dec_mag,dec_pha=dec_pha,vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()

################################################################################
########################### DIFFUSION TIMESTEPS ################################
################################################################################

hgt_ls = dec_mag.input_shape[1]
wdt_ls = dec_mag.input_shape[2]    

if args.LDM:
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

    # initialize the model in the memory of our GPU
    test_images = tf.ones((args.batch_size, hgt_ls, wdt_ls, args.encoded_size), dtype=tf.float32)
    test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
    k = unet(test_images, test_timestamps)

loss_fn = tf.losses.MeanSquaredError()


def sample(Z, Z_std=1.0, inference_timesteps=10):
    if args.LDM:
        # Create a range of inference steps that the output should be sampled at
        if args.DDIM:
            its = inference_timesteps
            inference_range = range(0, args.n_timesteps, args.n_timesteps // its)
        else:
            its = args.n_timesteps-1
            inference_range = range(1, args.n_timesteps)
        for index, i in enumerate(reversed(range(its))):
            t = np.expand_dims(inference_range[i], 0)

            pred_noise = unet(Z, t)
            if args.DDIM:
                Z = dm.ddim(Z, pred_noise, t, args.infer_sigma, alpha, alpha_bar)
            else:
                Z = dm.ddpm(Z, pred_noise, t, alpha, alpha_bar, beta)

    if args.VQ_encoder:
        vq_dict = vq_op(Z)
        Z = vq_dict['quantize']
    Z = tf.math.multiply_no_nan(Z,Z_std)
    Z2B_mag = dec_mag(Z, training=False)
    Z2B_pha = dec_pha(Z, training=False)
    Z2B_pha = tf.concat([tf.zeros_like(Z2B_pha[:,:,:,:,:1]),Z2B_pha],axis=-1)
    Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
    Z2B2A = IDEAL_op(Z2B, training=False)

    return Z2B, Z2B2A


# LS scaling factor
z_std = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32)

# checkpoint
if args.LDM:
    tl.Checkpoint(dict(unet=unet,z_std=z_std), py.join(args.experiment_dir, 'checkpoints_ldm')).restore()

# sample
sample_dir = py.join(output_dir, 'samples_ldm_testing', 'all')
wf_dir = py.join(output_dir, 'samples_ldm_testing', 'wf')
qmap_dir = py.join(output_dir, 'samples_ldm_testing', 'qmap')
mag_dir = py.join(output_dir, 'samples_ldm_testing', 'im_mag')
pha_dir = py.join(output_dir, 'samples_ldm_testing', 'im_phase')
py.mkdir(sample_dir)
py.mkdir(wf_dir)
py.mkdir(qmap_dir)
py.mkdir(mag_dir)
py.mkdir(pha_dir)

# main loop
for k in range(args.n_samples):
    Z = tf.random.normal((1,hgt_ls,wdt_ls,args.encoded_size), seed=args.seed, dtype=tf.float32)
    if args.VQ_encoder:
        Z2B, Z2B2A = sample(Z)
    else:
        Z2B, Z2B2A = sample(Z, z_std, inference_timesteps=args.infer_steps)

    fig, axs = plt.subplots(figsize=(20, 6), nrows=2, ncols=6)

    # Magnitude of recon MR images at each echo
    im_ech1 = np.squeeze(tf.complex(Z2B2A[:,0,:,:,0],Z2B2A[:,0,:,:,1]))
    im_ech2 = np.squeeze(tf.complex(Z2B2A[:,1,:,:,0],Z2B2A[:,1,:,:,1]))
    im_ech3 = np.squeeze(tf.complex(Z2B2A[:,2,:,:,0],Z2B2A[:,2,:,:,1]))
    im_ech4 = np.squeeze(tf.complex(Z2B2A[:,3,:,:,0],Z2B2A[:,3,:,:,1]))
    im_ech5 = np.squeeze(tf.complex(Z2B2A[:,4,:,:,0],Z2B2A[:,4,:,:,1]))
    im_ech6 = np.squeeze(tf.complex(Z2B2A[:,5,:,:,0],Z2B2A[:,5,:,:,1]))
    
    # Acquisitions in the first row
    acq_ech1 = axs[0,0].imshow(np.abs(im_ech1), cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
    axs[0,0].set_title('1st Echo')
    axs[0,0].axis('off')
    acq_ech2 = axs[0,1].imshow(np.abs(im_ech2), cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
    axs[0,1].set_title('2nd Echo')
    axs[0,1].axis('off')
    acq_ech3 = axs[0,2].imshow(np.abs(im_ech3), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,2].set_title('3rd Echo')
    axs[0,2].axis('off')
    acq_ech4 = axs[0,3].imshow(np.abs(im_ech4), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,3].set_title('4th Echo')
    axs[0,3].axis('off')
    acq_ech5 = axs[0,4].imshow(np.abs(im_ech5), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,4].set_title('5th Echo')
    axs[0,4].axis('off')
    acq_ech6 = axs[0,5].imshow(np.abs(im_ech6), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,5].set_title('6th Echo')
    axs[0,5].axis('off')

    # A2B maps in the second row
    if args.only_mag:
        w_m_aux = np.squeeze(Z2B[:,0,:,:,0])
        w_p_aux = np.squeeze(Z2B[:,1,:,:,0])
        f_m_aux = np.squeeze(Z2B[:,0,:,:,1])
        f_p_aux = np.squeeze(Z2B[:,1,:,:,1])
        r2_aux = np.squeeze(Z2B[:,0,:,:,2])
        field_aux = np.squeeze(Z2B[:,1,:,:,2])
    else:
        w_m_aux = np.squeeze(np.abs(tf.complex(Z2B[:,0,:,:,0],Z2B[:,0,:,:,1])))
        w_p_aux = np.squeeze(np.arctan2(Z2B[:,0,:,:,1],Z2B[:,0,:,:,0]))/np.pi
        f_m_aux = np.squeeze(np.abs(tf.complex(Z2B[:,1,:,:,0],Z2B[:,1,:,:,1])))
        f_p_aux = np.squeeze(np.arctan2(Z2B[:,1,:,:,1],Z2B[:,1,:,:,0]))/np.pi
        r2_aux = np.squeeze(Z2B[:,2,:,:,1])
        field_aux = np.squeeze(Z2B[:,2,:,:,0])
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

    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.8)
    plt.savefig(sample_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)

    # WF-imshow
    wf_fig, wf_ax = plt.subplots(figsize=(9,3))
    wf_all = np.concatenate([w_m_aux,f_m_aux],axis=1)
    wf_ax.imshow(wf_all, cmap='gray')
    wf_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(wf_dir+'/wf_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(wf_fig)

    # Show Q-maps 
    q_fig, q_axs = plt.subplots(figsize=(13,3), nrows=1, ncols=3)
    Fp_unet = q_axs[0].imshow(f_p_aux*3, cmap='twilight', vmin=-3, vmax=3)
    q_fig.colorbar(Fp_unet, ax=q_axs[0])
    q_axs[0].axis('off')
    r2_unet = q_axs[1].imshow(r2_aux*r2_sc, cmap='copper', vmin=0, vmax=r2_sc)
    q_fig.colorbar(r2_unet, ax=q_axs[1])
    q_axs[1].axis('off')
    field_unet = q_axs[2].imshow(field_aux*fm_sc, cmap='twilight', vmin=-fm_sc/2, vmax=fm_sc/2)
    q_fig.colorbar(field_unet, ax=q_axs[2])
    q_axs[2].axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(qmap_dir+'/qmap_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(q_fig)

    # Show all-echo magnitude
    all_echo = np.concatenate([im_ech1,im_ech2,im_ech3,im_ech4,im_ech5,im_ech6],axis=1)
    mag_fig, mag_ax = plt.subplots(figsize=(18,3))
    mag_ax.imshow(np.abs(all_echo), cmap='gray')
    mag_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(mag_dir+'/im_mag_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(mag_fig)

    # Show all-echo unwrapped phase
    pha_fig, pha_ax = plt.subplots(figsize=(21,3))
    im_pha = pha_ax.imshow(unwrap_phase(np.angle(all_echo))/np.pi, cmap='twilight', vmin=-4, vmax=4)
    pha_fig.colorbar(im_pha, ax=pha_ax)
    pha_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(pha_dir+'/im_phase_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(pha_fig)

