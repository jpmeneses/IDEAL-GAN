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
py.arg('--seed', type=int, default=0)
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

if not(hasattr(args,'rand_ne')):
	py.arg('--rand_ne', type=bool, default=False)
	ne_args = py.args()
	args.__dict__.update(ne_args.__dict__)

if not(hasattr(args,'div_decod')):
    py.arg('--div_decod', type=bool, default=False)
    dec_args = py.args()
    args.__dict__.update(dec_args.__dict__)


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
    nd = 3
else:
    nd = 1
if args.div_decod:
    if args.only_mag:
        nd = 2
    else:
        nd = 3
else:
    nd = 1
if args.only_mag:
    dec_mag = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_out),
                        filters=args.n_G_filters//nd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='relu',
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_pha = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_out-1),
                        filters=args.n_G_filters//(nd+1),
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation='tanh',
                        NL_self_attention=args.NL_SelfAttention
                        )
else:
    dec_w =  dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        filters=args.n_G_filters//nd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_f =  dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        filters=args.n_G_filters//nd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )
    dec_xi = dl.decoder(encoded_dims=args.encoded_size,
                        output_shape=(hgt,wdt,n_ch),
                        n_groups=args.n_groups_PM,
                        filters=args.n_G_filters//nd,
                        num_layers=args.n_downsamplings,
                        num_res_blocks=args.n_res_blocks,
                        output_activation=None,
                        NL_self_attention=args.NL_SelfAttention
                        )


if args.only_mag:
    IDEAL_op = wf.IDEAL_mag_Layer()
else:
    IDEAL_op = wf.IDEAL_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size,args.VQ_num_embed,args.VQ_commit_cost)

if args.only_mag:
    tl.Checkpoint(dict(dec_mag=dec_mag, dec_pha=dec_pha, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()
else:
    tl.Checkpoint(dict(dec_w=dec_w, dec_f=dec_f, dec_xi=dec_xi, vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample(Z,TE=None):
    if args.VQ_encoder:
        Z = vq_op.quantize(Z)
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
    # Reconstructed multi-echo images
    Z2B2A = IDEAL_op(Z2B)

    return Z2B, Z2B2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'Z2B')
py.mkdir(save_dir)

hls = hgt//(2**(args.n_downsamplings))
wls = wdt//(2**(args.n_downsamplings))
z_shape = (1,hls,wls,args.encoded_size)

TE = wf.gen_TEvar(6,orig=False)
# Z = tf.random.normal(z_shape,seed=1,dtype=tf.float32)

for k in range(args.n_samples):
    if args.VQ_encoder:
        Z = tf.random.uniform(z_shape[:-1],minval=0,maxval=args.VQ_num_embed,seed=args.seed,dtype=tf.int32)
    else:
        Z = tf.random.normal(z_shape,seed=args.seed,dtype=tf.float32)
    Z2B, Z2B2A = sample(Z)

    if args.only_mag:
        w_m_aux = np.squeeze(Z2B[:,0,:,:,0])
        w_p_aux = np.squeeze(Z2B[:,1,:,:,0])
        f_m_aux = np.squeeze(Z2B[:,0,:,:,1])
        f_p_aux = np.squeeze(Z2B[:,1,:,:,1])
        r2_aux = np.squeeze(Z2B[:,0,:,:,2])
        field_aux = np.squeeze(Z2B[:,1,:,:,2])
    else:
        w_m_aux = np.squeeze(np.abs(tf.complex(Z2B[:,0,:,:,0],Z2B[:,0,:,:,1])))
        w_p_aux = np.squeeze(np.arctan2(Z2B[:,0,:,:,1],Z2B[:,0,:,:,0]))/np.pi
        f_m_aux = np.squeeze(np.abs(tf.complex(Z2B[:,1,:,:,0],Z2B[:,1,:,:,1])))
        f_p_aux = np.squeeze(np.arctan2(Z2B[:,1,:,:,1],Z2B[:,1,:,:,0]))/np.pi
        r2_aux = np.squeeze(Z2B[:,2,:,:,1])
        field_aux = np.squeeze(Z2B[:,2,:,:,0])

    im_ech1 = np.squeeze(np.abs(tf.complex(Z2B2A[:,0,:,:,0],Z2B2A[:,0,:,:,1])))
    im_ech2 = np.squeeze(np.abs(tf.complex(Z2B2A[:,1,:,:,0],Z2B2A[:,1,:,:,1])))
    im_ech3 = np.squeeze(np.abs(tf.complex(Z2B2A[:,2,:,:,0],Z2B2A[:,2,:,:,1])))
    im_ech4 = np.squeeze(np.abs(tf.complex(Z2B2A[:,3,:,:,0],Z2B2A[:,3,:,:,1])))
    im_ech5 = np.squeeze(np.abs(tf.complex(Z2B2A[:,4,:,:,0],Z2B2A[:,4,:,:,1])))
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
    acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                            interpolation='none', vmin=0, vmax=1)
    axs[0,2].set_title('3rd Echo')
    axs[0,2].axis('off')
    acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                            interpolation='none', vmin=0, vmax=1)
    axs[0,3].set_title('4th Echo')
    axs[0,3].axis('off')
    acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                            interpolation='none', vmin=0, vmax=1)
    axs[0,4].set_title('5th Echo')
    axs[0,4].axis('off')
    acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                            interpolation='none', vmin=0, vmax=1)
    axs[0,5].set_title('6th Echo')
    axs[0,5].axis('off')

    # Z2B maps in the second row
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

    fig.suptitle('TE1/dTE: '+str([TE[0,0,0].numpy(),np.mean(np.diff(TE, axis=1))]), fontsize=16)

    # plt.show()
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.8)
    plt.savefig(save_dir+'/sample'+str(k).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)
