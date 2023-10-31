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
py.arg('--n_samples', type=int, default=50)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

if not(hasattr(args,'VQ_num_embed')):
	py.arg('--VQ_num_embed', type=bool, default=256)
	py.arg('--VQ_commit_cost', type=int, default=0.5)
	VQ_args = py.args()
	args.__dict__.update(VQ_args.__dict__)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

hgt,wdt,n_ch = 192,192,2

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

dec= dl.decoder(encoded_dims=args.encoded_size,
                output_shape=(hgt,wdt,n_ch),
                multi_echo=False,
                filters=args.n_G_filters,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                output_activation=None,
                NL_self_attention=args.NL_SelfAttention
                )


IDEAL_op = wf.IDEAL_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)

tl.Checkpoint(dict(dec=dec, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample(Z,TE=None):
	if args.VQ_encoder:
		Z = vq_op.quantize(Z)
	# Z2B2A Cycle
	Z2A = dec(Z, training=False)
	return Z2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'Z2A')
py.mkdir(save_dir)

hls = hgt//(2**(args.n_downsamplings))
wls = wdt//(2**(args.n_downsamplings))
z_shape = (2,hls,wls,args.encoded_size)

# TE = wf.gen_TEvar(args.n_echoes,orig=False)
# Z = tf.random.normal(z_shape,seed=1,dtype=tf.float32)

for k in range(args.n_samples):
	if args.VQ_encoder:
		Z = tf.random.uniform(z_shape[:-1],minval=0,maxval=args.VQ_num_embed,seed=0,dtype=tf.int32)
	else:
		Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
	Z2A = sample(Z)

	fig, axs = plt.subplots(figsize=(5,3), ncols=2)

	# Magnitude of recon MR images at each echo
	im_ech1 = np.squeeze(np.abs(tf.complex(Z2A[0,:,:,0], Z2A[0,:,:,1])))
	im_ech2 = np.squeeze(np.abs(tf.complex(Z2A[1,:,:,0], Z2A[1,:,:,1])))

	# Acquisitions in the first row
	acq_ech1 = axs[0].imshow(im_ech1, cmap='gist_earth',
							interpolation='none', vmin=0, vmax=1)
	axs[0].set_title('Gen No.1')
	axs[0].axis('off')
	acq_ech2 = axs[1].imshow(im_ech2, cmap='gist_earth',
							interpolation='none', vmin=0, vmax=1)
	axs[1].set_title('Gen No.2')
	axs[1].axis('off')

	# plt.show()
	plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
	tl.make_space_above(axs,topmargin=0.8)
	plt.savefig(save_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
	plt.close(fig)