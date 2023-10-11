import functools

import random
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import DLlib as dl
import pylib as py
import wflib as wf
import data

py.arg('--dataset', default='sinGAN')
py.arg('--K_sc', type=int, default=3)
py.arg('--n_D_layers', type=int, default=4)
py.arg('--epochs', type=int, default=3000)
py.arg('--epoch_ckpt', type=int, default=500)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--beta_2', type=float, default=0.9)
py.arg('--data_aug_p', type=float, default=0.4)
py.arg('--DC_loss_weight', type=float, default=100.0)
py.arg('--R1_reg_weight', type=float, default=0.2)
py.arg('--R2_reg_weight', type=float, default=0.2)
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# A2A_pool = data.ItemPool(args.pool_size)
fm_sc = 300.0
r2_sc = 200.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/' #'../datasets/'

dataset_hdf5_4 = 'Attilio_GC_384_complex_2D_nMsk.hdf5'
acqs, out_maps = data.load_hdf5(dataset_dir, dataset_hdf5_4, 12, end=20,
								acqs_data=True, te_data=False)

################################################################################
############################# DATASET PARTITIONS ###############################
################################################################################

trainX = np.squeeze(acqs[9,:,:,:])
hgt,wdt,n_ch = np.shape(trainX)

trainY = np.squeeze(out_maps[9,:,:,:])
_,_,n_out = np.shape(trainY)

# Re-scale images
K_max = 3
hgt_mx = hgt//(2**K_max)
wdt_mx = wdt//(2**K_max)
trainX_mx = np.zeros((hgt_mx,wdt_mx,n_ch),dtype='float32')
trainY_mx = np.zeros((hgt_mx,wdt_mx,n_out),dtype='float32')
if args.K_sc < K_max:
	hgt_ds = hgt//(2**args.K_sc)
	wdt_ds = wdt//(2**args.K_sc)
	trainX_k = np.zeros((hgt_ds,wdt_ds,n_ch),dtype='float32')
	trainY_k = np.zeros((hgt_ds,wdt_ds,n_out),dtype='float32')
for ch in range(n_ch):
	trainX_mx[:,:,ch] = Image.fromarray(trainX[:,:,ch], mode="F").resize((hgt_mx,wdt_mx),resample=Image.LANCZOS)
	if args.K_sc < K_max:
		trainX_k[:,:,ch] = Image.fromarray(trainX[:,:,ch], mode="F").resize((hgt_ds,wdt_ds),resample=Image.LANCZOS)
trainX_mx = np.expand_dims(trainX_mx, axis=0)
for ot in range(n_out):
	trainY_mx[:,:,ot] = Image.fromarray(trainY[:,:,ot], mode="F").resize((hgt_mx,wdt_mx),resample=Image.LANCZOS)
	if args.K_sc < K_max:
		trainY_k[:,:,ot] = Image.fromarray(trainY[:,:,ot], mode="F").resize((hgt_ds,wdt_ds),resample=Image.LANCZOS)
trainY_mx = np.expand_dims(trainY_mx, axis=0)
if args.K_sc < K_max:
	trainX_k = np.expand_dims(trainX_k, axis=0)
	trainY_k = np.expand_dims(trainY_k, axis=0)
else:
	trainX_k = trainX_mx
	trainY_k = trainY_mx

print('Original Dimensions:', hgt, wdt)
print('Max K Dimensions:', hgt_mx, wdt_mx)
print('Echoes:', acqs.shape[1])
print('Num. Channels:', n_ch)
print('Num. Outputs:', n_out)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

# G_0 = dl.sGAN(input_shape=(hgt//(2**0),wdt//(2**0),n_ch),gen_mode=True)
# G_1 = dl.sGAN(input_shape=(hgt//(2**1),wdt//(2**1),n_ch),gen_mode=True)
# G_2 = dl.sGAN(input_shape=(hgt//(2**2),wdt//(2**2),n_ch),gen_mode=True)
# G_3 = dl.sGAN(input_shape=(hgt//(2**3),wdt//(2**3),n_ch),gen_mode=True)

G_0 = dl.UNet(input_shape=(hgt//(2**0),wdt//(2**0),n_ch),n_out=n_out,filters=32,self_attention=True)
G_1 = dl.UNet(input_shape=(hgt//(2**1),wdt//(2**1),n_ch),n_out=n_out,filters=32,self_attention=True)
G_2 = dl.UNet(input_shape=(hgt//(2**2),wdt//(2**2),n_ch),n_out=n_out,filters=32,self_attention=True)
G_3 = dl.UNet(input_shape=(hgt//(2**3),wdt//(2**3),n_ch),n_out=n_out,filters=32,self_attention=True)

D_0 = dl.sGAN(input_shape=(None,None,n_ch))
D_1 = dl.sGAN(input_shape=(None,None,n_ch))
D_2 = dl.sGAN(input_shape=(None,None,n_ch))
D_3 = dl.sGAN(input_shape=(None,None,n_ch))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')
cycle_loss_fn = tf.losses.MeanSquaredError()

G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)
D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1, beta_2=args.beta_2)

IDEAL_op = wf.IDEAL_Layer()
B_reshape = keras.layers.Lambda(lambda x:tf.concat([tf.expand_dims(x[:,:,:,:2],1),
													tf.expand_dims(x[:,:,:,2:4],1),
													tf.expand_dims(x[:,:,:,4:],1)],axis=1))
A_reshape = keras.layers.Lambda(lambda x: tf.reshape(tf.transpose(x,perm=[0,2,3,1,4]),[x.shape[0],x.shape[2],x.shape[3],-1]))

@tf.function
def train_G(A, B, G, D):
    with tf.GradientTape() as t:
    	A2B = G(A, training=True)
    	# A_res = tf.where(A!=0.0,A_res,0.0)
    	DC_loss = cycle_loss_fn(B, A2B)

    	A2B = B_reshape(A2B, training=False)
    	A2B2A = IDEAL_op(A2B, training=False)
    	A2B2A = A_reshape(A2B2A, training=False)
    	
    	A2A_d_logits = D(A2B2A, training=False)
    	D_loss = g_loss_fn(A2A_d_logits[-1])

    	G_loss = args.DC_loss_weight * DC_loss + D_loss
        
    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return A2B2A, {'D_loss': D_loss, 'DC_loss': DC_loss}

@tf.function
def train_D(A_real, A_fake, D):
    with tf.GradientTape() as t:
        A_d_logits = D(A_real, training=True)
        A2A_d_logits = D(A_fake, training=True)
        
        A_d_loss, A2A_d_loss = d_loss_fn(A_d_logits[-1], A2A_d_logits[-1])

        D_A_r1 = gan.R1_regularization(functools.partial(D, training=True), A_real)
        D_A_r2 = gan.R1_regularization(functools.partial(D, training=True), A_fake)

        D_loss = (A_d_loss + A2A_d_loss) + (D_A_r1 * args.R1_reg_weight) + (D_A_r2 * args.R2_reg_weight)

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))
    return {'D_loss': A_d_loss + A2A_d_loss, 'D_A_r1': D_A_r1, 'D_A_r2': D_A_r2}

def train_step(A, B, A_ref, G, D):
    A_res, G_loss_dict = train_G(A, B, G, D)

    # cannot autograph `A2B_pool`
    # A_res = A2A_pool(A_res)
    D_loss_dict = train_D(A_ref, A_res, D)

    return A_res, G_loss_dict, D_loss_dict

def upscale(A, G):
	A2B = G(A, training=False)
	A2B = B_reshape(A2B, training=False)
	A_res = IDEAL_op(A2B, training=False)
	A_res = A_reshape(A_res, training=False)
	# A_res = tf.where(A!=0.0,A_res,0.0)
	A_res = np.squeeze(A_res, axis=0)
	hgt_ups, wdt_ups = (2*A_res.shape[-3],2*A_res.shape[-2])
	A_ups = np.zeros((hgt_ups,wdt_ups,n_ch),dtype='float32')
	for ch in range(n_ch):
		A_ups[:,:,ch] = Image.fromarray(A_res[:,:,ch], mode="F").resize((hgt_ups,wdt_ups),resample=Image.LANCZOS)
	return np.expand_dims(A_ups, axis=0)

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_0=G_0,
                                G_1=G_1,
                                G_2=G_2,
                                G_3=G_3,
                                D_0=D_0,
                                D_1=D_1,
                                D_2=D_2,
                                D_3=D_3),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=2)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'k%01d' % args.K_sc))

# sample
sample_dir = py.join(output_dir, 'samples_training', 'k%01d' % args.K_sc)
py.mkdir(sample_dir)

if args.K_sc <= 2:
	trainX_mx = upscale(trainX_mx, G_3)
if args.K_sc <= 1:
	trainX_mx = upscale(trainX_mx, G_2)
if args.K_sc <= 0:
	trainX_mx = upscale(trainX_mx, G_1)
print('Upsampled input shape:', trainX_mx.shape)
A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX_mx,trainX_k,trainY_k)).batch(1)

# main loop
for ep in range(args.epochs):
	if ep < ep_cnt:
		continue
	# update epoch counter
	ep_cnt.assign_add(1)
	# train for an epoch
	for A, A_ref, B in A_B_dataset:
		p = np.random.rand()
		if p <= args.data_aug_p:
			A_2 = tf.concat([A,A_ref,B], axis=-1)
			A_2 = tf.image.rot90(A_2,k=np.random.randint(3))
			A_2 = tf.image.random_flip_left_right(A_2)
			A_2 = tf.image.random_flip_up_down(A_2)
			A = A_2[:,:,:,:n_ch]
			A_ref = A_2[:,:,:,n_ch:(2*n_ch)]
			B = A_2[:,:,:,(2*n_ch):]
		if args.K_sc == 0:
			A_res, G_loss_dict, D_loss_dict = train_step(A, B, A_ref, G_0, D_0)
		elif args.K_sc == 1:
			A_res, G_loss_dict, D_loss_dict = train_step(A, B, A_ref, G_1, D_1)
		elif args.K_sc == 2:
			A_res, G_loss_dict, D_loss_dict = train_step(A, B, A_ref, G_2, D_2)
		elif args.K_sc == 3:
			A_res, G_loss_dict, D_loss_dict = train_step(A, B, A_ref, G_3, D_3)
		with train_summary_writer.as_default():
			tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
			tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')
		if ep % 300 == 0:
			fig, axs = plt.subplots(figsize=(9, 3), ncols=3)
			A_abs = np.squeeze(np.abs(tf.complex(A[:,:,:,0],A[:,:,:,1])))
			acq_in = axs[0].imshow(A_abs, cmap='gist_earth', vmin=0, vmax=1)
			axs[0].set_title('Input')
			axs[0].axis('off')
			A_res_abs = np.squeeze(np.abs(tf.complex(A_res[:,:,:,0],A_res[:,:,:,1])))
			acq_out = axs[1].imshow(A_res_abs, cmap='gist_earth', vmin=0, vmax=1)
			axs[1].set_title('Output')
			axs[1].axis('off')
			A_ref_abs = np.squeeze(np.abs(tf.complex(A_ref[:,:,:,0],A_ref[:,:,:,1])))
			acq_ref = axs[2].imshow(A_ref_abs, cmap='gist_earth', vmin=0, vmax=1)
			axs[2].set_title('Ref')
			axs[2].axis('off')
			plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
			tl.make_space_above(axs,topmargin=0.8)
			plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()), bbox_inches = 'tight', pad_inches = 0)
			plt.close(fig)
	if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
		checkpoint.save(ep)