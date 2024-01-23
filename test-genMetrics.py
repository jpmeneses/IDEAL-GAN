import itertools
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
py.arg('--LDM', type=bool, default=False)
py.arg('--te_input', type=bool, default=False)
py.arg('--val_batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

if not(hasattr(args,'data_size')):
    py.arg('--data_size', type=int, default=192, choices=[192,384])
    ds_args = py.args()
    args.__dict__.update(ds_args.__dict__)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

fm_sc = 300.0
r2_sc = 200.0

dataset_dir = '../datasets/'
dataset_hdf5_2 = 'INTA_GC_' + str(args.data_size) + '_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_2, 12, MEBCRN=True, mag_and_phase=args.only_mag)

len_dataset,ne,hgt,wdt,n_ch = valX.shape
A_dataset_val = tf.data.Dataset.from_tensor_slices(valX)
A_dataset_val = A_dataset_val.batch(args.val_batch_size).shuffle(len_dataset)

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
    nfd = [a//nd for a in filt_list]
    nfd2 = [a//(nd+1) for a in filt_list]
else:
    nfd = args.n_G_filters//nd
    nfd2= args.n_G_filters//(nd+1)
if args.only_mag:
    dec_mag = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,3),
                        filters=nfd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='relu',
                        output_initializer='he_normal',
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_pha = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,2),
                        filters=nfd2,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='tanh',
                        NL_self_attention=args.NL_SelfAttention
                        )
    tl.Checkpoint(dict(dec_mag=dec_mag,dec_pha=dec_pha), py.join(args.experiment_dir, 'checkpoints')).restore()
    hgt_ls = dec_mag.input_shape[1]
    wdt_ls = dec_mag.input_shape[2]
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
    tl.Checkpoint(dict(dec_w=dec_w,dec_f=dec_f,dec_xi=dec_xi), py.join(args.experiment_dir, 'checkpoints')).restore()
    hgt_ls = dec_w.input_shape[1]
    wdt_ls = dec_w.input_shape[2]
if args.LDM:
    # create our unet model
    unet = dl.denoise_Unet(dim=args.n_ldm_filters, dim_mults=(1,2,4), channels=args.encoded_size)
    z_std = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)
    # Initiate unet
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

get_features = dl.get_features((ne,hgt,wdt,n_ch))

if args.only_mag:
    IDEAL_op = wf.IDEAL_mag_Layer()
else:
    IDEAL_op = wf.IDEAL_Layer()

def encode(A):
	A2Z = enc(A, training=True)
	return A2Z


def sample(Z,denoise=False,TE=None):
    if denoise:
        for i in range(args.n_timesteps-1):
            t = np.expand_dims(np.array(args.n_timesteps-i-1, np.int32), 0)
            pred_noise = unet(Z, t)
            Z = dm.ddpm(Z, pred_noise, t, alpha, alpha_bar, beta)
    # Z2B2A Cycle
    if args.only_mag:
        Z2B_mag = dec_mag(Z, training=True)
        Z2B_pha = dec_pha(Z, training=True)
        Z2B_pha = tf.concat([tf.zeros_like(Z2B_pha[:,:,:,:,:1]),Z2B_pha],axis=-1)
        Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
    else:
        Z2B_w = dec_w(Z, training=False)
        Z2B_f = dec_f(Z, training=False)
        Z2B_xi= dec_xi(Z, training=False)
        Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
    # Reconstructed multi-echo images
    Z2B2A = IDEAL_op(Z2B)

    return Z2B2A


synth_features = []
real_features = []

mmd_scores = []

ms_ssim_scores = []
ssim_scores = []

fid = dl.FID()
mmd = dl.MMD()

for A in A_dataset_val:
    # Generate some synthetic images using the defined model
    z_shape = (A.shape[0],hgt_ls,wdt_ls,args.encoded_size)
    Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
    Z2B, Z2B2A = sample(Z, denoise=args.LDM)

    # Get the features for the real data
    real_eval_feats = get_features(A)
    real_features.append(real_eval_feats)

    # Get the features for the synthetic data
    synth_eval_feats = get_features(Z2B2A)
    synth_features.append(synth_eval_feats)

    # SSIM metrics for pairs of synthetic data within batch
    idx_pairs = list(itertools.combinations(range(A.shape[0]), 2))
    for idx_a, idx_b in idx_pairs:
        ms_ssim_scores.append(tf.image.ssim_multiscale(Z2B2A[idx_a]+1.0, Z2B2A[idx_b]+1.0, 2))
        ssim_scores.append(tf.image.ssim(Z2B2A[idx_a]+1.0, Z2B2A[idx_b]+1.0, 2))

    # Auto-encode real image
    A2Z = encode(A)
    A2Z2A = sample(A2Z)

    # Compute MMD
    # mmd.reset_state()
    mmd_scores.append(mmd(A, A2Z2A))

	
synth_features = tf.concat(synth_features,axis=0)
real_features = tf.concat(real_features,axis=0)

fid_res = fid(synth_features, real_features)
print(f"FID Score: {fid_res.numpy():.4f}")

mmd_res = mmd_scores[-1] / len_dataset
print(f"MMD Score: {mmd_res.numpy():.4f}")

ms_ssim_scores = tf.concat(ms_ssim_scores,axis=0)
print(f"MS-SSIM Score: {tf.reduce_mean(ms_ssim_scores).numpy():.4f} +- {tf.math.reduce_std(ms_ssim_scores).numpy():.4f}")

ssim_scores = tf.concat(ssim_scores,axis=0)
print(f"SSIM Score: {tf.reduce_mean(ssim_scores).numpy():.4f} +- {tf.math.reduce_std(ssim_scores).numpy():.4f}")