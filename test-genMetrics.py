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

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--te_input', type=bool, default=False)
py.arg('--val_batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
fm_sc = 300.0
r2_sc = 2*np.pi*fm_sc

dataset_dir = '../datasets/'
dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_2, ech_idx, MEBCRN=True)

len_dataset,ne,hgt,wdt,n_ch = valX.shape
A_dataset_val = tf.data.Dataset.from_tensor_slices(valX)
A_dataset_val = A_dataset_val.batch(args.val_batch_size).shuffle(len_dataset)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(args.n_echoes,hgt,wdt,n_ch),
    				encoded_dims=args.encoded_size,
                    filters=args.n_G_filters,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    NL_self_attention=args.NL_SelfAttention)
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

get_features = dl.get_features((ne,hgt,wdt,n_ch))

IDEAL_op = wf.IDEAL_Layer(args.n_echoes,MEBCRN=True)

tl.Checkpoint(dict(dec_w=dec_w, dec_f=dec_f, dec_xi=dec_xi), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample(Z,TE=None):
	# Z2B2A Cycle
	Z2B_w = dec_w(Z, training=False)
	Z2B_f = dec_f(Z, training=False)
	Z2B_xi= dec_xi(Z, training=False)
	Z2B = tf.concat([Z2B_w,Z2B_f,Z2B_xi],axis=1)
	# Water/fat magnitudes
	Z2B_WF_real = tf.concat([Z2B_w[:,0,:,:,:1],Z2B_f[:,0,:,:,:1]],axis=-1)
	Z2B_WF_imag = tf.concat([Z2B_w[:,0,:,:,1:],Z2B_f[:,0,:,:,1:]],axis=-1)
	Z2B_WF_abs = tf.abs(tf.complex(Z2B_WF_real,Z2B_WF_imag))
	Z2B_abs = tf.concat([Z2B_WF_abs,tf.squeeze(Z2B_xi,axis=1)],axis=-1)
	# Reconstructed multi-echo images
	Z2B2A = IDEAL_op(Z2B)

	return Z2B_abs, Z2B2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'Z2B')
py.mkdir(save_dir)

hls = hgt//(2**(args.n_downsamplings))
wls = wdt//(2**(args.n_downsamplings))

synth_features = []
real_features = []

for A in A_dataset_val:
    # Generate some synthetic images using the defined model
    z_shape = (A.shape[0],hls,wls,args.encoded_size)
    Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
    Z2B, Z2B2A = sample(Z)

    # Get the features for the real data
    real_eval_feats = get_features(A)
    real_features.append(real_eval_feats)

    # Get the features for the synthetic data
    synth_eval_feats = get_features(Z2B2A)
    synth_features.append(synth_eval_feats)

	
synth_features = tf.stack(synth_features)
real_features = tf.stack(real_features)

fid = dl.FIDMetric()
fid_res = fid(synth_features, real_features)

print(f"FID Score: {fid_res.item():.4f}")