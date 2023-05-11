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

hgt,wdt = 192,192
ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0


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
def sample(Z,TE=None):
	indices =tf.concat([tf.zeros((Z.shape[0],hgt,wdt,4),dtype=tf.int32),
                        tf.ones((Z.shape[0],hgt,wdt,1),dtype=tf.int32),
                        2*tf.ones((Z.shape[0],hgt,wdt,1),dtype=tf.int32)],axis=-1)
	# Z2B2A Cycle
	Z2B = dec(Z, training=False)
	# Split A2B param maps
	Z2B_WF,Z2B_R2,Z2B_FM = tf.dynamic_partition(Z2B,indices,num_partitions=3)
	Z2B_WF = tf.reshape(Z2B_WF,Z2B[:,:,:,:4].shape)
	Z2B_R2 = tf.reshape(Z2B_R2,Z2B[:,:,:,:1].shape)
	Z2B_FM = tf.reshape(Z2B_FM,Z2B[:,:,:,:1].shape)
	# Correct R2 scaling
	Z2B_R2 = 0.5*Z2B_R2 + 0.5
	Z2B = tf.concat([Z2B_WF,Z2B_R2,Z2B_FM],axis=-1)
	# Water/fat magnitudes
	Z2B_WF_real = Z2B_WF[:,:,:,0::2]
	Z2B_WF_imag = Z2B_WF[:,:,:,1::2]
	Z2B_WF_abs = tf.abs(tf.complex(Z2B_WF_real,Z2B_WF_imag))
	Z2B_abs = tf.concat([Z2B_WF_abs,Z2B_R2,Z2B_FM],axis=-1)
	# Reconstructed multi-echo images
	Z2B2A = wf.IDEAL_model(Z2B,args.n_echoes,te=TE)

	return Z2B_abs, Z2B2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'Z2B')
py.mkdir(save_dir)

Z_shape = (args.n_samples,enc.output_shape[1],enc.output_shape[2],enc.output_shape[3])
Z = tf.random.uniform(Z_shape)

TE = wf.gen_TEvar(args.n_echoes,args.n_samples,orig=False)

Z2B, Z2B2A = sample(Z,TE)

for k in range(args.n_samples):
	w_aux = np.squeeze(Z2B[k,:,:,0])
	f_aux = np.squeeze(Z2B[k,:,:,1])
	r2_aux = np.squeeze(Z2B[k,:,:,2])
	field_aux = np.squeeze(Z2B[k,:,:,3])

	im_ech1 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,0],Z2B2A[k,:,:,1])))
	im_ech2 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,2],Z2B2A[k,:,:,3])))
	if args.n_echoes >= 3:
	    im_ech3 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,4],Z2B2A[k,:,:,5])))
	if args.n_echoes >= 4:
	    im_ech4 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,6],Z2B2A[k,:,:,7])))
	if args.n_echoes >= 5:
	    im_ech5 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,8],Z2B2A[k,:,:,9])))
	if args.n_echoes >= 6:
	    im_ech6 = np.squeeze(np.abs(tf.complex(Z2B2A[k,:,:,10],Z2B2A[k,:,:,11])))

	fig, axs = plt.subplots(figsize=(20, 6), nrows=2, ncols=6)

	# Acquisitions in the first row
	acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
	axs[0,0].set_title('1st Echo')
	axs[0,0].axis('off')
	acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                          interpolation='none', vmin=0, vmax=1)
	axs[0,1].set_title('2nd Echo')
	axs[0,1].axis('off')
	if args.n_echoes >= 3:
		acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
		axs[0,2].set_title('3rd Echo')
		axs[0,2].axis('off')
	else:
		fig.delaxes(axs[0,2])
	if args.n_echoes >= 4:
		acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
		axs[0,3].set_title('4th Echo')
		axs[0,3].axis('off')
	else:
		fig.delaxes(axs[0,3])
	if args.n_echoes >= 5:
		acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
		axs[0,4].set_title('5th Echo')
		axs[0,4].axis('off')
	else:
		fig.delaxes(axs[0,4])
	if args.n_echoes >= 6:
		acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                              interpolation='none', vmin=0, vmax=1)
		axs[0,5].set_title('6th Echo')
		axs[0,5].axis('off')
	else:
		fig.delaxes(axs[0,5])

	# Z2B maps in the second row
	W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                            interpolation='none', vmin=0, vmax=1)
	fig.colorbar(W_ok, ax=axs[1,1])
	axs[1,1].axis('off')

	F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                            interpolation='none', vmin=0, vmax=1)
	fig.colorbar(F_ok, ax=axs[1,2])
	axs[1,2].axis('off')

	r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                            interpolation='none', vmin=0, vmax=r2_sc)
	fig.colorbar(r2_ok, ax=axs[1,3])
	axs[1,3].axis('off')

	field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
	fig.colorbar(field_ok, ax=axs[1,4])
	axs[1,4].axis('off')
	fig.delaxes(axs[1,0])
	fig.delaxes(axs[1,5])

	fig.suptitle('TE1/dTE: '+str([TE[k,0].numpy(),np.mean(np.diff(TE[k,:]))]), fontsize=16)

	# plt.show()
	plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
	tl.make_space_above(axs,topmargin=0.8)
	plt.savefig(save_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
	plt.close(fig)
