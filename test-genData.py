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


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
fm_sc = 300.0
r2_sc = 200.0
hgt,wdt,n_ch = 192,192,2


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
z_shape = (1,hls,wls,args.encoded_size)

TE = wf.gen_TEvar(args.n_echoes,orig=False)

for k in range(args.n_samples):
	Z = tf.random.normal(z_shape,seed=0,dtype=tf.float32)
	Z2B, Z2B2A = sample(Z)

	w_aux = np.squeeze(Z2B[:,:,:,0])
	f_aux = np.squeeze(Z2B[:,:,:,1])
	r2_aux = np.squeeze(Z2B[:,:,:,3])
	field_aux = np.squeeze(Z2B[:,:,:,2])

	im_ech1 = np.squeeze(np.abs(tf.complex(Z2B2A[:,0,:,:,0],Z2B2A[:,0,:,:,1])))
	im_ech2 = np.squeeze(np.abs(tf.complex(Z2B2A[:,1,:,:,0],Z2B2A[:,1,:,:,1])))
	if args.n_echoes >= 3:
	    im_ech3 = np.squeeze(np.abs(tf.complex(Z2B2A[:,2,:,:,0],Z2B2A[:,2,:,:,1])))
	if args.n_echoes >= 4:
	    im_ech4 = np.squeeze(np.abs(tf.complex(Z2B2A[:,3,:,:,0],Z2B2A[:,3,:,:,1])))
	if args.n_echoes >= 5:
	    im_ech5 = np.squeeze(np.abs(tf.complex(Z2B2A[:,4,:,:,0],Z2B2A[:,4,:,:,1])))
	if args.n_echoes >= 6:
	    im_ech6 = np.squeeze(np.abs(tf.complex(Z2B2A[:,5,:,:,0],Z2B2A[:,5,:,:,1])))

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

	fig.suptitle('TE1/dTE: '+str([TE[0,0].numpy(),np.mean(np.diff(TE[0,:]))]), fontsize=16)

	# plt.show()
	plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
	tl.make_space_above(axs,topmargin=0.8)
	plt.savefig(save_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
	plt.close(fig)
