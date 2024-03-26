import numpy as np
import tensorflow as tf

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
py.arg('--ds_filename', default='LDM_ds')
py.arg('--te_input', type=bool, default=False)
py.arg('--DDIM', type=bool, default=False)
py.arg('--infer_steps', type=int, default=10)
py.arg('--infer_sigma', type=float, default=0.0)
py.arg('--batch_size', type=int, default=1)
py.arg('--n_samples', type=int, default=2000)
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

if not(hasattr(args,'div_decod')):
    py.arg('--div_decod', type=bool, default=False)
    dec_args = py.args()
    args.__dict__.update(dec_args.__dict__)

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

# create our unet model
unet = dl.denoise_Unet(dim=args.n_ldm_filters, dim_mults=(1,2,4), channels=args.encoded_size)

if args.only_mag:
    IDEAL_op = wf.IDEAL_mag_Layer()
else:
    IDEAL_op = wf.IDEAL_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size, args.VQ_num_embed, args.VQ_commit_cost)

if args.only_mag:
    tl.Checkpoint(dict(dec_mag=dec_mag,dec_pha=dec_pha,vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()
else:
    tl.Checkpoint(dict(dec_w=dec_w,dec_f=dec_f,dec_xi=dec_xi,vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()

################################################################################
########################### DIFFUSION TIMESTEPS ################################
################################################################################

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
if args.only_mag:
    hgt_ls = dec_mag.input_shape[1]
    wdt_ls = dec_mag.input_shape[2]
else:
    hgt_ls = dec_w.input_shape[1]
    wdt_ls = dec_w.input_shape[2]
test_images = tf.ones((args.batch_size, hgt_ls, wdt_ls, args.encoded_size), dtype=tf.float32)
test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
k = unet(test_images, test_timestamps)

loss_fn = tf.losses.MeanSquaredError()

def sample(Z, Z_std=1.0, inference_timesteps=10, ns=0):
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
    if args.only_mag:
        Z2B_mag = dec_mag(Z, training=False)
        Z2B_pha = dec_pha(Z, training=False)
        Z2B_pha = tf.concat([tf.zeros_like(Z2B_pha[:,:,:,:,:1]),Z2B_pha],axis=-1)
        Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
    else:
        Z2B_w = dec_w(Z, training=False)
        Z2B_f = dec_f(Z, training=False)
        Z2B_xi= dec_xi(Z, training=False)
        Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
    Z2B2A = IDEAL_op(Z2B, training=False)

    return Z2B, Z2B2A

# LS scaling factor
z_std = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

# checkpoint
tl.Checkpoint(dict(unet=unet,z_std=z_std), py.join(args.experiment_dir, 'checkpoints_ldm')).restore()

# sample
ds_dir = 'tfrecord'
ds_filename = args.ds_filename
py.mkdir(ds_dir)
writer = tf.io.TFRecordWriter(py.join(ds_dir,ds_filename))

# main loop
for k in range(args.n_samples):
    Z = tf.random.normal((args.batch_size,hgt_ls,wdt_ls,args.encoded_size), seed=args.seed, dtype=tf.float32)
    if args.VQ_encoder:
        Z2B, Z2B2A = sample(Z)
    else:
        Z2B, Z2B2A = sample(Z, z_std, inference_timesteps=args.infer_steps, ns=k)

    for i in range(Z2B.shape[0]):
        acqs_i = Z2B2A[i,...]
        out_maps_i = Z2B[i,...]
        features = {'acqs': _bytes_feature(tf.io.serialize_tensor(acqs_i)),
                    'acq_shape': _int64_feature(list(acqs_i.shape)),
                    'out_maps': _bytes_feature(tf.io.serialize_tensor(out_maps_i)),
                    'out_shape': _int64_feature(list(out_maps_i.shape))}

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())

writer.close()