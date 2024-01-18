import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tf2lib as tl
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf
import data

from PIL import Image

# Save a GIF using logged images
def save_gif(img_list, mode="RGB", path="", interval=200):
    # Transform images from [-1,1] to [0, 255]
    imgs = []
    for im in img_list:
        im = im.numpy()
        if mode=='L': # In this case, input range is [0,1]
            im = im * 255
        else:
            im = (im + 2) * 127.5/2
        im = np.clip(im, 0, 255).astype(np.uint8)
        im = Image.fromarray(im, mode=mode)
        imgs.append(im)
    
    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs,
             save_all=True, duration=interval, loop=0)


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='output/GAN-100')
py.arg('--scheduler', default='linear', choices=['linear','cosine'])
py.arg('--n_timesteps', type=int, default=200)
py.arg('--beta_start', type=float, default=0.0001)
py.arg('--beta_end', type=float, default=0.02)
py.arg('--s_value', type=float, default=8e-3)
py.arg('--n_samples', type=int, default=50)
ldm_args = py.args()

output_dir = py.join(ldm_args.experiment_dir)
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

if not(hasattr(args,'ls_mean_activ')):
    py.arg('--ls_mean_activ', default='leaky_relu', choices=['leaky_relu','relu','tanh','None'])
    lsa_args = py.args()
    args.__dict__.update(lsa_args.__dict__)

if hasattr(args,'n_G_filt_list'):
    if len(args.n_G_filt_list) > 0:
        filt_list = [int(a_i) for a_i in args.n_G_filt_list.split(',')]
    else:
        filt_list = list()

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'

dataset_hdf5_2 = 'JGalgani_GC_' + str(args.data_size) + '_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_2, end=args.n_samples, MEBCRN=True, mag_and_phase=args.only_mag)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Input and output dimensions (validations data)
print('Validation input shape:',valX.shape)
print('Validation output shape:',valY.shape)

# Overall dataset statistics
len_dataset,_,hgt,wdt,n_ch = np.shape(valX)
if args.only_mag:
    _,_,_,_,n_out = np.shape(valY)
else:
    _,n_out,_,_,_ = np.shape(valY)

print('Image Dimensions:', hgt, wdt)
print('Num. Output Maps:',n_out)

A_dataset = tf.data.Dataset.from_tensor_slices(valX)
A_dataset = A_dataset.batch(1)

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
                ls_mean_activ=args.ls_mean_activ,
                NL_self_attention=args.NL_SelfAttention
                )
dec_mag = dl.decoder(encoded_dims=args.encoded_size,
                    output_shape=(hgt,wdt,n_out),
                    filters=nfd,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation='relu',
                    NL_self_attention=args.NL_SelfAttention
                    )

tl.Checkpoint(dict(enc=enc, dec_mag=dec_mag), py.join(args.experiment_dir, 'checkpoints')).restore()

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

@tf.function
def encode(A, Z_std=1.0):
    A2Z = enc(A, training=False)
    return tf.math.divide_no_nan(A2Z,Z_std)

@tf.function
def decode(Z, Z_std=1.0):
    Z = tf.math.multiply_no_nan(Z,Z_std)
    Z2B = dec_mag(Z, training=False)
    Z2B_PDFF = tf.math.divide_no_nan(Z2B[:,0,:,:,1],Z2B[:,0,:,:,0]+Z2B[:,0,:,:,1])
    # Z2B_PDFF = Z2B[:,0,:,:,0]
    return Z2B, Z2B_PDFF

# LS scaling factor
z_std = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

# checkpoint
checkpoint_ldm = tl.Checkpoint(dict(z_std=z_std),
                               py.join(output_dir, 'checkpoints_ldm'),
                               max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint_ldm.restore().assert_existing_objects_matched()
    print('Scaling Factor:',z_std.numpy())
except Exception as e:
    print(e)

# sample
sample_dir = py.join(output_dir, 'samples_forward_diff')
py.mkdir(sample_dir)

i = 0    

for A in A_dataset:
    print('SHAPE',A.shape)
    A2Z = encode(A, z_std)
    _, A2B_PDFF = decode(A2Z, z_std)
    # A2B_PDFF = tf.reduce_sum(tf.square(A[:,0,:,:,:]),axis=-1,keepdims=False)
    img_list = [tf.squeeze(A2Z,axis=0)]
    map_list = [tf.squeeze(A2B_PDFF,axis=0)]
    for beta_t in beta:
        noise = tf.random.normal(A2Z.shape, dtype=tf.float32)
        A2Z = np.sqrt(1-beta_t) * A2Z + beta_t * noise
        img_list.append(tf.squeeze(A2Z,axis=0))
        _, A2B_PDFF = decode(A2Z, z_std)
        map_list.append(tf.squeeze(A2B_PDFF,axis=0))

    save_gif(([img_list[0]] * 100) + img_list + ([img_list[-1]] * 100), path=py.join(sample_dir,'sample-%03d.gif' % i), interval=20)
    save_gif(([map_list[0]] * 100) + map_list + ([map_list[-1]] * 100), mode='L', path=py.join(sample_dir,'map-%03d.gif' % i), interval=20)
    i+=1

img_list = []
for beta_t in beta:
    noise = tf.random.normal(A2Z.shape, dtype=tf.float32)
    img_list.append(tf.squeeze(noise,axis=0))
save_gif(([img_list[0]] * 100) + img_list + ([img_list[-1]] * 100), path=py.join(sample_dir,'sample-%03d.gif' % i), interval=20)