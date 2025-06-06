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
py.arg('--MEBCRN', type=bool, default=True)
py.arg('--DDIM', type=bool, default=False)
py.arg('--infer_steps', type=int, default=10)
py.arg('--infer_sigma', type=float, default=0.0)
py.arg('--batch_size', type=int, default=1)
py.arg('--n_samples', type=int, default=2000)
py.arg('--num_classes_1', type=int, default=3)
py.arg('--gen_class_1', type=int, default=None)
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

if not(hasattr(args,'unwrap')):
    py.arg('--unwrap', type=bool, default=True)
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
unet =  dl.denoise_Unet(dim=args.n_ldm_filters,
                        dim_mults=(1,2,4), 
                        channels=args.encoded_size,
                        num_classes=args.num_classes_1,
                        in_res=dec_mag.input_shape[1])

IDEAL_op = wf.IDEAL_mag_Layer()

vq_op = dl.VectorQuantizer(args.encoded_size, args.VQ_num_embed, args.VQ_commit_cost)

tl.Checkpoint(dict(dec_mag=dec_mag,dec_pha=dec_pha,vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()

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
hgt_ls = dec_mag.input_shape[1]
wdt_ls = dec_mag.input_shape[2]

test_images = tf.ones((args.batch_size, hgt_ls, wdt_ls, args.encoded_size), dtype=tf.float32)
test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
test_label = np.random.randint(4, size=(args.batch_size,), dtype=np.int32)
k = unet(test_images, test_timestamps, test_label)

loss_fn = tf.losses.MeanSquaredError()

def sample(Z, label, Z_std=1.0, inference_timesteps=10, ns=0):
    # Create a range of inference steps that the output should be sampled at
    if args.DDIM:
        its = inference_timesteps
        inference_range = range(0, args.n_timesteps, args.n_timesteps // its)
    else:
        its = args.n_timesteps-1
        inference_range = range(1, args.n_timesteps)
    for index, i in enumerate(reversed(range(its))):
        t = np.expand_dims(inference_range[i], 0)

        pred_noise = unet(Z, t, label)
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
    Z2B_pha = tf.concat([Z2B_pha[:,:,:,:,:1],Z2B_pha],axis=-1)
    Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
    Z2B2A = IDEAL_op(Z2B, training=False)
    if not(args.MEBCRN):
        if args.unwrap:
            c_pha = 3
        else:
            c_pha = 1
        Z2B_W_r = Z2B_mag[:,0,:,:,:1] * tf.math.cos(c_pha*Z2B_pha[:,0,:,:,1:2]*np.pi)
        Z2B_W_i = Z2B_mag[:,0,:,:,:1] * tf.math.sin(c_pha*Z2B_pha[:,0,:,:,1:2]*np.pi)
        Z2B_F_r = Z2B_mag[:,0,:,:,1:2]* tf.math.cos(c_pha*Z2B_pha[:,0,:,:,1:2]*np.pi)
        Z2B_F_i = Z2B_mag[:,0,:,:,1:2]* tf.math.sin(c_pha*Z2B_pha[:,0,:,:,1:2]*np.pi)
        Z2B_r2 = Z2B_mag[:,0,:,:,2:]
        Z2B_fm = Z2B_pha[:,0,:,:,2:]
        Z2B = tf.concat([Z2B_W_r,Z2B_W_i,Z2B_F_r,Z2B_F_i,Z2B_r2,Z2B_fm],axis=-1)
        
        Re_rho = tf.transpose(Z2B2A[:,:,:,:,0], perm=[0,2,3,1])
        Im_rho = tf.transpose(Z2B2A[:,:,:,:,1], perm=[0,2,3,1])
        zero_fill = tf.zeros_like(Re_rho)
        re_stack = tf.stack([Re_rho,zero_fill],4)
        re_aux = tf.reshape(re_stack,[Z.shape[0],hgt,wdt,2*Z2B2A.shape[1]])
        im_stack = tf.stack([zero_fill,Im_rho],4)
        im_aux = tf.reshape(im_stack,[Z.shape[0],hgt,wdt,2*Z2B2A.shape[1]])
        Z2B2A = re_aux + im_aux

    return Z2B, Z2B2A

# LS scaling factor
z_std = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

# checkpoint
tl.Checkpoint(dict(unet=unet,z_std=z_std), py.join(args.experiment_dir, 'checkpoints_ldm')).restore()

# sample
method_prefix = 'm999'
pre_filename_1 = 'PDFF_p00_'
pre_filename_2 = 'R2s_p00_'
pre_filename_3 = 'multiecho_p00_'
end_filename = '_gen'

ds_dir = 'tfrecord'
ds_filename = args.ds_filename + '_' + str(args.n_samples)
py.mkdir(ds_dir)
writer = tf.io.TFRecordWriter(py.join(ds_dir,ds_filename))

# main loop
n_vol = 0
for k in range(args.n_samples//args.batch_size):
    Z = tf.random.normal((args.batch_size,hgt_ls,wdt_ls,args.encoded_size), seed=args.seed, dtype=tf.float32)
    if args.gen_class_1 is None:
        Lv= np.random.randint(args.num_classes_1, size=(args.batch_size,), dtype=np.int32)
    else:
        Lv= np.array([args.gen_class_1]*args.batch_size)
    if args.VQ_encoder:
        Z2B, Z2B2A = sample(Z, Lv)
    else:
        Z2B, Z2B2A = sample(Z, Lv, z_std, inference_timesteps=args.infer_steps, ns=k)

    for i in range(Z2B.shape[0]):
        volun_name = 'v' + str(n_vol+i).zfill(4)
        # Write PDFF maps
        X1 = tf.squeeze(Z2B[i,0,:,:,1]/(Z2B[i,0,:,:,0]+Z2B[i,0,:,:,1]))
        X1 = tf.clip_by_value(X1,0.0,1.0)
        filename_1 = pre_filename_1 + volun_name + end_filename
        path_1 = py.join(args.experiment_dir,"out_dicom",'Volunteer-'+volun_name[1:],'Method-'+method_prefix[1:],'PDFF')
        py.mkdir(path_1)
        image2d_1 = X1.numpy()
        ds1 = data.gen_ds(n_vol+i, method_prefix)
        data.write_dicom(ds1, image2d_1, path_1, filename_1, 0, 1)
        # Write R2s maps
        X2 = tf.squeeze(Z2B[i,0,:,:,2] )
        X2 = tf.clip_by_value(X2,0.0,1.0)
        filename_2 = pre_filename_2 + volun_name + end_filename
        path_2 = py.join(args.experiment_dir,"out_dicom",'Volunteer-'+volun_name[1:],'Method-'+method_prefix[1:],'R2s')
        py.mkdir(path_2)
        image2d_2 = X2.numpy()
        ds2 = data.gen_ds(n_vol+i, method_prefix, R2s=True)
        data.write_dicom(ds2, image2d_2, path_2, filename_2, 0, 1)
        # Write multi-echo magnitude images
        X3 = tf.math.sqrt(tf.reduce_sum(tf.square(Z2B2A[i,...]),axis=-1)) 
        X3 = tf.squeeze(X3)
        X3 = tf.clip_by_value(X3,0.0,1.0)
        filename_3 = pre_filename_3 + volun_name + end_filename
        path_3 = py.join(args.experiment_dir,"out_dicom",'Volunteer-'+volun_name[1:],'Method-'+method_prefix[1:],'MultiEcho')
        py.mkdir(path_3)
        ds3 = data.gen_ds(n_vol+i, method_prefix)
        for j in range(X3.shape[0]):
            image2d_3 = tf.squeeze(X3[j,...])
            data.write_dicom(ds3, image2d_3.numpy(), path_3, filename_3, j, X3.shape[0])
        
        acqs_i = Z2B2A[i,...]
        out_maps_i = Z2B[i,...]
        features = {'acqs': data._bytes_feature(tf.io.serialize_tensor(acqs_i)),
                    'acq_shape': data._int64_feature(list(acqs_i.shape)),
                    'out_maps': data._bytes_feature(tf.io.serialize_tensor(out_maps_i)),
                    'out_shape': data._int64_feature(list(out_maps_i.shape))}

        example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(example.SerializeToString())
    n_vol += Z2B.shape[0]

writer.close()