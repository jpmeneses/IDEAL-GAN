import functools
import os

import random
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
from sklearn.decomposition import PCA

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
py.arg('--n_samples', type=int, default=10)
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
acqs, out_maps = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx, MEBCRN=True)

len_dataset,_,hgt,wdt,n_ch = np.shape(acqs)
_,_,_,n_out = np.shape(out_maps)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

if args.G_model == 'encod-decod':
    enc= dl.encoder(input_shape=(args.n_echoes,hgt,wdt,n_ch),
    				encoded_size=args.encoded_size,
                    filters=args.n_G_filters,
                    )
    dec= dl.decoder(input_shape=(args.encoded_size),
                    output_shape=(hgt,wdt,n_out),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
    G_A2B = tf.keras.Sequential()
    G_A2B.add(enc)
    G_A2B.add(dec)
else:
    raise(NameError('Unrecognized Generator Architecture'))

tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample(A):
	indices =tf.concat([tf.zeros((A.shape[0],hgt,wdt,4),dtype=tf.int32),
                        tf.ones((A.shape[0],hgt,wdt,1),dtype=tf.int32),
                        2*tf.ones((A.shape[0],hgt,wdt,1),dtype=tf.int32)],axis=-1)
	A2Z = enc(A, training=False)
	A2Z2B = dec(A2Z, training=False)
	# Split A2B param maps
	A2Z2B_WF,A2Z2B_R2,A2Z2B_FM = tf.dynamic_partition(A2Z2B,indices,num_partitions=3)
	A2Z2B_WF = tf.reshape(A2Z2B_WF,A2Z2B[:,:,:,:4].shape)
	A2Z2B_R2 = tf.reshape(A2Z2B_R2,A2Z2B[:,:,:,:1].shape)
	A2Z2B_FM = tf.reshape(A2Z2B_FM,A2Z2B[:,:,:,:1].shape)
	# Correct R2 scaling
	A2Z2B_R2 = 0.5*A2Z2B_R2 + 0.5
	A2Z2B = tf.concat([A2Z2B_WF,A2Z2B_R2,A2Z2B_FM],axis=-1)
	return A2Z, A2Z2B

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2Z')
py.mkdir(save_dir)

idxs = np.random.choice([a for a in range(len_dataset)],args.n_samples)
cont = 0
for i in idxs:
	A = acqs[i:i+1,:,:,:,:]
	A2Z_i, A2Z2B_i = sample(A)
	if cont==0:
		A2Z, A2Z2B = A2Z_i, A2Z2B_i
	else:
		A2Z = tf.concat([A2Z,A2Z_i],axis=0)
		A2Z2B = tf.concat([A2Z2B,A2Z2B_i],axis=0)
	cont += 1
print('A2Z shape:',A2Z.shape)
print('A2Z2B shape:',A2Z2B.shape)

# mapper = umap.UMAP().fit(A2Z)
# print('Mapper shape:',mapper.shape)
# umap.plot.points(mapper)#, labels=digits.target)
# plt.show()

pca = PCA(n_components = 10)
encoded_imgs = pca.fit_transform(A2Z)
print(encoded_imgs.shape)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(encoded_imgs[:,0],encoded_imgs[:,1], s=8, cmap='tab10')

def onclick(event):
    global flag
    if event.xdata is None or event.ydata is None:
        return
    ix, iy = np.round(event.xdata,-2), np.round(event.ydata,-2)
    for idx in range(args.n_samples):
    	if np.round(encoded_imgs[idx,0],-2)==ix and np.round(encoded_imgs[idx,1],-2)==iy:
    		# print('Sample encountered!')
    		A2Z2B_W = tf.squeeze(tf.complex(A2Z2B[idx,:,:,0],A2Z2B[idx,:,:,1]))
    		ax[1].imshow(np.abs(A2Z2B_W), cmap='gray')
    		plt.draw()
    	#else:
    		#print('No sample at this point')

# button_press_event
# motion_notify_event
cid = fig.canvas.mpl_connect('motion_notify_event', onclick)


plt.show() 
