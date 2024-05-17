import functools
import itertools

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

import tf2lib as tl
import tf2gan as gan
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--dataset', type=str, default='JGalgani', choices=['multiTE','3ech','JGalgani','phantom_1p5','phantom_3p0'])
py.arg('--n_echoes', type=int, default=6)
py.arg('--remove_imag', type=bool, default=False)
py.arg('--LDM', type=bool, default=False)
py.arg('--infer_steps', type=int, default=10)
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
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

if not(hasattr(args,'unwrap')):
    py.arg('--unwrap', type=bool, default=True)
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
ech_idx = 2*args.n_echoes

dataset_dir = '../datasets/'
if args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
    dataset_hdf5 = args.dataset + '_RF_192_128_complex_2D.hdf5'
else:
    dataset_hdf5 = args.dataset + '_GC_' + str(args.data_size) + '_complex_2D.hdf5'

if args.dataset == 'JGalgani':
    num_slice_list = [0,21,20,24,24,21,24,21,21,24,21,21,22,27,23,22,20,24,21,21,22,20]
    rnc = True
elif args.dataset == 'multiTE':
    ini_idxs = [0,84,204,300,396,484,580,680,776,848,932,1028, 1100,1142,1190,1232,1286,1334,1388,1460]
    delta_idxs = [21,24,24,24,22,24,25,24,18,21,24,18, 21,24,21,18,16,18,24,21]
    # First Patient
    if args.TE1 == 0.0014 and args.dTE == 0.0022:
        k_idxs = [(0,1),(2,3)]
    elif args.TE1 == 0.0013 and args.dTE == 0.0023:
        k_idxs = [(0,1),(3,4)]
    else:
        k_idxs = [(0,2)]
    for k in k_idxs:
        custom_list = [a for a in range(ini_idxs[0]+k[0]*delta_idxs[0],ini_idxs[0]+k[1]*delta_idxs[0])]
    # Rest of the patients
    for i in range(1,len(ini_idxs)):
        if (i<=11) and args.TE1 == 0.0013 and args.dTE == 0.0022:
            k_idxs = [(0,1),(2,3)]
        elif (i<=11) and args.TE1 == 0.0014 and args.dTE == 0.0022:
            k_idxs = [(0,1),(3,4)]
        elif (i==1) and args.TE1 == 0.0013 and args.dTE == 0.0023:
            k_idxs = [(0,1),(4,5)]
        elif (i==15 or i==16) and args.TE1 == 0.0013 and args.dTE == 0.0023:
            k_idxs = [(0,1),(2,3)]
        elif (i>=17) and args.TE1 == 0.0013 and args.dTE == 0.0024:
            k_idxs = [(0,1),(2,3)]
        else:
            k_idxs = [(0,2)]
        for k in k_idxs:
            custom_list += [a for a in range(ini_idxs[i]+k[0]*delta_idxs[i],ini_idxs[i]+k[1]*delta_idxs[i])]
else:
    num_slice_list = None
    rnc = False

if args.dataset == 'JGalgani' or args.dataset == '3ech':
    valX, valY=data.load_hdf5(dataset_dir,dataset_hdf5,ech_idx,num_slice_list=num_slice_list,remove_non_central=rnc,
                                acqs_data=True,te_data=False,remove_zeros=True,MEBCRN=True,mag_and_phase=args.only_mag,unwrap=args.unwrap)
    TEs = wf.gen_TEvar(args.n_echoes, bs=valX.shape[0], orig=True)
elif args.dataset == 'multiTE':
    valX, valY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, custom_list=custom_list,
                                        acqs_data=True,te_data=True,remove_zeros=False,MEBCRN=True,mag_and_phase=args.only_mag,unwrap=args.unwrap)
else:
    valX, valY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, acqs_data=True, 
                                        te_data=True,remove_zeros=True,MEBCRN=True,mag_and_phase=args.only_mag,unwrap=args.unwrap)
if args.dataset == 'multiTE':
    valX, valY, TEs = data.group_TEs(valX,valY,TEs,TE1=args.TE1,dTE=args.dTE,MEBCRN=True)

len_dataset,ne,hgt,wdt,n_ch = valX.shape
if args.only_mag:
    _,_,_,_,n_out = np.shape(valY)
else:
    _,n_out,_,_,_ = np.shape(valY)

print('Acquisition Dimensions:', hgt, wdt)
print('Echoes:', ne)
print('Output Maps:', n_out)

A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,TEs,valY))
A_B_dataset_val = A_B_dataset_val.batch(args.val_batch_size)

z_std = tf.Variable(initial_value=1.0, trainable=False, dtype=tf.float32)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

nd = 2
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
                NL_self_attention=args.NL_SelfAttention)
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

if args.LDM:
    unet = dl.denoise_Unet(dim=args.n_ldm_filters, dim_mults=(1,2,4), channels=args.encoded_size)
    # Initiate unet
    hgt_ls = dec_mag.input_shape[1]
    wdt_ls = dec_mag.input_shape[2]
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

IDEAL_op = wf.IDEAL_mag_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)

tl.Checkpoint(dict(enc=enc, dec_mag=dec_mag, dec_pha=dec_pha, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()


# @tf.function
def sample(A,Z_std,TE=None):
    # Turn complex-valued CSE-MR image into only-real
    if args.remove_imag:
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
    A2Z2B_mag = dec_mag(A2Z, training=False)
    A2Z2B_pha = dec_pha(A2Z, training=False)
    A2Z2B_pha = tf.concat([A2Z2B_pha[:,:,:,:,:1],A2Z2B_pha],axis=-1)
    A2B = tf.concat([A2Z2B_mag,A2Z2B_pha],axis=1)
    # Reconstructed multi-echo images
    A2B2A = IDEAL_op(A2B,te=TE)

    return A2B, A2B2A

# run
k = 0
# sample
sample_dir = py.join(args.experiment_dir, 'samples_testing', 'all')
wf_dir = py.join(args.experiment_dir, 'samples_testing', 'wf')
qmap_dir = py.join(args.experiment_dir, 'samples_testing', 'qmap')
mag_dir = py.join(args.experiment_dir, 'samples_testing', 'im_mag')
pha_dir = py.join(args.experiment_dir, 'samples_testing', 'im_phase')
py.mkdir(sample_dir)
py.mkdir(wf_dir)
py.mkdir(qmap_dir)
py.mkdir(mag_dir)
py.mkdir(pha_dir)

ms_ssim_scores = []
ssim_scores = []
apd_scores = []

APD_loss_fn = gan.AbsolutePhaseDisparity()

# main loop
for A, TE, B in A_B_dataset_val:
    # Get only-magnitude latent space
    A2B, A2B2A = sample(A,z_std,TE)

    fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

    # Magnitude of recon MR images at each echo
    im_ech1 = np.squeeze(tf.complex(A[:,0,:,:,0],A[:,0,:,:,1]))
    im_ech2 = np.squeeze(tf.complex(A[:,1,:,:,0],A[:,1,:,:,1]))
    im_ech3 = np.squeeze(tf.complex(A[:,2,:,:,0],A[:,2,:,:,1]))
    if args.n_echoes > 3:
        im_ech4 = np.squeeze(tf.complex(A[:,3,:,:,0],A[:,3,:,:,1]))
    else:
        im_ech4 = np.zeros_like(im_ech3)
    if args.n_echoes > 4:
        im_ech5 = np.squeeze(tf.complex(A[:,4,:,:,0],A[:,4,:,:,1]))
    else:
        im_ech5 = np.zeros_like(im_ech3)
    if args.n_echoes > 5:
        im_ech6 = np.squeeze(tf.complex(A[:,5,:,:,0],A[:,5,:,:,1]))
    else:
        im_ech6 = np.zeros_like(im_ech3)
    
    recon_ech1 = np.squeeze(tf.complex(A2B2A[:,0,:,:,0],A2B2A[:,0,:,:,1]))
    recon_ech2 = np.squeeze(tf.complex(A2B2A[:,1,:,:,0],A2B2A[:,1,:,:,1]))
    recon_ech3 = np.squeeze(tf.complex(A2B2A[:,2,:,:,0],A2B2A[:,2,:,:,1]))
    recon_ech4 = np.squeeze(tf.complex(A2B2A[:,3,:,:,0],A2B2A[:,3,:,:,1]))
    recon_ech5 = np.squeeze(tf.complex(A2B2A[:,4,:,:,0],A2B2A[:,4,:,:,1]))
    recon_ech6 = np.squeeze(tf.complex(A2B2A[:,5,:,:,0],A2B2A[:,5,:,:,1]))
    
    # Acquisitions in the first row
    acq_ech1 = axs[0,0].imshow(np.abs(recon_ech1), cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
    axs[0,0].set_title('1st Echo')
    axs[0,0].axis('off')
    acq_ech2 = axs[0,1].imshow(np.abs(recon_ech2), cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
    axs[0,1].set_title('2nd Echo')
    axs[0,1].axis('off')
    acq_ech3 = axs[0,2].imshow(np.abs(recon_ech3), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,2].set_title('3rd Echo')
    axs[0,2].axis('off')
    acq_ech4 = axs[0,3].imshow(np.abs(recon_ech4), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,3].set_title('4th Echo')
    axs[0,3].axis('off')
    acq_ech5 = axs[0,4].imshow(np.abs(recon_ech5), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,4].set_title('5th Echo')
    axs[0,4].axis('off')
    acq_ech6 = axs[0,5].imshow(np.abs(recon_ech6), cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
    axs[0,5].set_title('6th Echo')
    axs[0,5].axis('off')

    # A2B maps in the second row
    w_m_aux = np.squeeze(A2B[:,0,:,:,0])
    w_p_aux = np.squeeze(A2B[:,1,:,:,0])
    f_m_aux = np.squeeze(A2B[:,0,:,:,1])
    f_p_aux = np.squeeze(A2B[:,1,:,:,1])
    r2_aux = np.squeeze(A2B[:,0,:,:,2])
    field_aux = np.squeeze(A2B[:,1,:,:,2])

    if args.only_mag:
        # Save APD loss
        if args.unwrap:
            B_WF = tf.concat([B[:,:1,:,:,:2],3*B[:,1:,:,:,:2]],axis=1)
            A2B_WF = tf.concat([A2B[:,:1,:,:,:2],3*A2B[:,1:,:,:,:2]],axis=1)
        else:
            B_WF = B[:,:,:,:,:2]
            A2B_WF = A2B[:,:,:,:,:2]
        apd_scores.append(APD_loss_fn(B_WF, A2B_WF))

        wn_m_aux = np.squeeze(B[:,0,:,:,0])
        wn_p_aux = np.squeeze(B[:,1,:,:,0])
        fn_m_aux = np.squeeze(B[:,0,:,:,1])
        fn_p_aux = np.squeeze(B[:,1,:,:,1])
        r2n_aux = np.squeeze(B[:,0,:,:,2])
        fieldn_aux = np.squeeze(B[:,1,:,:,2])

    else:
        B_wf_mag = tf.abs(tf.complex(B[:,:2,:,:,:1],B[:,:2,:,:,1:]))
        B_wf_mag = tf.transpose(B_wf_mag, perm=[0,4,2,3,1])
        B_wf_phase = tf.math.atan2(B[:,:2,:,:,1:],B[:,:2,:,:,:1])
        B_wf_phase = tf.transpose(B_wf_phase, perm=[0,4,2,3,1])
        B_cplx = tf.concat([B_wf_mag,B_wf_phase], axis=1)

        if args.unwrap:
            A2B_cplx = tf.concat([A2B[:,:1,:,:,:2],3*A2B[:,1:,:,:,:2]],axis=1)
        else:
            A2B_cplx = A2B[:,:,:,:,:2]

        # Save APD loss
        apd_scores.append(APD_loss_fn(B_cplx, A2B_cplx))

        wn_m_aux = np.squeeze(np.abs(tf.complex(B[:,0,:,:,0],B[:,0,:,:,1])))
        wn_p_aux = np.squeeze(np.arctan2(B[:,0,:,:,1],B[:,0,:,:,0]))/np.pi
        fn_m_aux = np.squeeze(np.abs(tf.complex(B[:,1,:,:,0],B[:,1,:,:,1])))
        fn_p_aux = np.squeeze(np.arctan2(B[:,1,:,:,1],B[:,1,:,:,0]))/np.pi
        r2n_aux = np.squeeze(B[:,2,:,:,1])
        fieldn_aux = np.squeeze(B[:,2,:,:,0])

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

    W_gt =  axs[2,0].imshow(wn_m_aux, cmap='bone',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_gt, ax=axs[2,0])
    axs[2,0].axis('off')

    Wp_gt =  axs[2,1].imshow(wn_p_aux, cmap='twilight',
                            interpolation='none', vmin=-1, vmax=1)
    fig.colorbar(Wp_gt, ax=axs[2,1])
    axs[2,1].axis('off')

    F_gt =  axs[2,2].imshow(fn_m_aux, cmap='pink',
                            interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_gt, ax=axs[2,2])
    axs[2,2].axis('off')

    Fp_gt = axs[2,3].imshow(fn_p_aux, cmap='twilight',
                            interpolation='none', vmin=-1, vmax=1)
    fig.colorbar(Fp_gt, ax=axs[2,3])
    axs[2,3].axis('off')

    r2_gt = axs[2,4].imshow(r2n_aux*r2_sc, cmap='copper',
                            interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_gt, ax=axs[2,4])
    axs[2,4].axis('off')

    field_gt =  axs[2,5].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_gt, ax=axs[2,5])
    axs[2,5].axis('off')

    fig.suptitle('TE1/dTE: '+str([TE[0,0,0].numpy(),np.mean(np.diff(TE,axis=1))]), fontsize=16)
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.8)
    plt.savefig(sample_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)

    # WF-imshow
    wf_fig, wf_ax = plt.subplots(figsize=(9,6))
    wf_ok = np.concatenate([w_m_aux,f_m_aux],axis=1)
    wf_gt = np.concatenate([wn_m_aux,fn_m_aux],axis=1)
    wf_all = np.concatenate([wf_ok,wf_gt],axis=0)
    wf_ax.imshow(wf_all, cmap='gray')
    wf_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(wf_dir+'/wf_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(wf_fig)

    # Show Q-maps 
    q_fig, q_axs = plt.subplots(figsize=(13,6), nrows=2, ncols=3)
    Fp_unet = q_axs[0,0].imshow(f_p_aux*3, cmap='twilight', vmin=-3, vmax=3)
    q_fig.colorbar(Fp_unet, ax=q_axs[0,0])
    q_axs[0,0].axis('off')
    r2_unet = q_axs[0,1].imshow(r2_aux*r2_sc, cmap='copper', vmin=0, vmax=r2_sc)
    q_fig.colorbar(r2_unet, ax=q_axs[0,1])
    q_axs[0,1].axis('off')
    field_unet = q_axs[0,2].imshow(field_aux*fm_sc, cmap='twilight', vmin=-fm_sc/2, vmax=fm_sc/2)
    q_fig.colorbar(field_unet, ax=q_axs[0,2])
    q_axs[0,2].axis('off')
    Fp_gt = q_axs[1,0].imshow(fn_p_aux*3, cmap='twilight', vmin=-3, vmax=3)
    q_fig.colorbar(Fp_gt, ax=q_axs[1,0])
    q_axs[1,0].axis('off')
    r2_gt = q_axs[1,1].imshow(r2n_aux*r2_sc, cmap='copper', vmin=0, vmax=r2_sc)
    q_fig.colorbar(r2_gt, ax=q_axs[1,1])
    q_axs[1,1].axis('off')
    field_unet = q_axs[1,2].imshow(fieldn_aux*fm_sc, cmap='twilight', vmin=-fm_sc/2, vmax=fm_sc/2)
    q_fig.colorbar(field_gt, ax=q_axs[1,2])
    q_axs[1,2].axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(qmap_dir+'/qmap_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(q_fig)

    # Show all-echo magnitude
    orig_echo = np.concatenate([im_ech1,im_ech2,im_ech3,im_ech4,im_ech5,im_ech6],axis=1)
    recon_echo = np.concatenate([recon_ech1,recon_ech2,recon_ech3,recon_ech4,recon_ech5,recon_ech6],axis=1)
    all_echo = np.concatenate([orig_echo,recon_echo],axis=0)
    mag_fig, mag_ax = plt.subplots(figsize=(18,6))
    mag_ax.imshow(np.abs(all_echo), cmap='gray')
    mag_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(mag_dir+'/im_mag_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(mag_fig)

    # Show all-echo unwrapped phase
    pha_fig, pha_ax = plt.subplots(figsize=(21,6))
    orig_pha = unwrap_phase(np.angle(orig_echo))/np.pi
    recon_pha = unwrap_phase(np.angle(recon_echo))/np.pi
    im_pha = pha_ax.imshow(np.concatenate([orig_pha,recon_pha],axis=0), cmap='twilight', vmin=-4, vmax=4)
    pha_fig.colorbar(im_pha, ax=pha_ax)
    pha_ax.axis('off')
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    plt.savefig(pha_dir+'/im_phase_sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(pha_fig)

    k+=1

apd_scores = tf.concat(apd_scores,axis=0)
print(f"WF APD loss: {tf.reduce_mean(apd_scores).numpy():.4f} +- {tf.math.reduce_std(apd_scores).numpy():.4f}")