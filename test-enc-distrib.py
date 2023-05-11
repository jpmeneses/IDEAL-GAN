import functools

import random
import numpy as np
import matplotlib.pyplot as plt

import DLlib as dl
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import wflib as wf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--te_input', type=bool, default=False)
py.arg('--n_samples', type=int, default=50)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
acqs, out_maps = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx)

len_dataset,hgt,wdt,d_ech = np.shape(acqs)
_,_,_,n_out = np.shape(out_maps)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(hgt,wdt,ech_idx),
                    filters=args.n_G_filters,
                    te_shape=(args.n_echoes,),
                    )
    dec= dl.decoder(input_shape=enc.output_shape[1:],
                    n_out=6,
                    te_shape=(args.n_echoes,),
                    self_attention=args.D1_SelfAttention)
    G_A2B = tf.keras.Sequential()
    G_A2B.add(enc)
    G_A2B.add(dec)
else:
    raise(NameError('Unrecognized Generator Architecture'))

tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample(A):
	# Z2B2A Cycle
	A2Z = enc(A, training=False)

	return A2Z

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2Z')
py.mkdir(save_dir)

A_idxs = np.random.choice([a for a in range(len_dataset)],args.n_samples)
A = acqs[A_idxs,:,:,:]
Z = sample(A)

Z_shape = (args.n_samples,enc.output_shape[1],enc.output_shape[2],enc.output_shape[3])
Z = tf.random.uniform(Z_shape,seed=0)
counts, bins = np.histogram(Z,bins=100)

plt.stairs(counts, bins)
plt.savefig(save_dir+'/Z-histogram-n'+str(args.n_samples).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)