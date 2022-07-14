import functools

import random
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import tqdm

import DLlib as dl
import pylib as py
import tensorflow as tf
import tf2lib as tl
import tf2gan as gan
import wflib as wf

import data
from keras_unet.models import custom_unet

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--batch_size', type=int, default=4)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                   excel                                    =
# ==============================================================================
workbook = xlsxwriter.Workbook(py.join(args.experiment_dir, 'uncertainty.xlsx'))
ws_MAE = workbook.add_worksheet('Var')
ws_MAE.write(0,0,'Water')
ws_MAE.write(0,1,'Fat')
ws_MAE.write(0,2,'PDFF')
ws_MAE.write(0,3,'R2*')
ws_MAE.write(0,4,'FieldMap')

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../MRI-Datasets/'
dataset_hdf5_1 = 'UNet-JGalgani/JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

dataset_hdf5_4 = 'UNet-Volunteers/Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                    acqs_data=True, te_data=False,
                                    complex_data=False)

################################################################################
############################# DATASET PARTITIONS ###############################
################################################################################

n1_div = 248
n4_div = 434

testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Sample to plot
samples = [12,36,128,173,260,379,478,527,648]
testX = testX[samples,:,:,:]
testY = testY[samples,:,:,:]

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(testX)
_,_,_,n_out = np.shape(testY)
echoes = int(d_ech/2)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_ds_test = tf.data.Dataset.from_tensor_slices(testX)
A_ds_test.batch(args.batch_size)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

# with tf.device('/cpu:0'):
if args.G_model == 'multi-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,d_ech),
                                filters=args.n_filters,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                                filters=args.n_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        n_out = 4
    else:
        n_out = 2
    G_A2B = custom_unet(input_shape=(hgt,wdt,d_ech),
                        num_classes=n_out,
                        dropout=0.01,
                        use_dropout_on_upsampling=True,
                        use_attention=args.D1_SelfAttention,
                        filters=args.n_filters)

elif args.G_model == 'MEBCRN':
    if args.out_vars == 'WF-PM':
        n_out = 4
    else:
        n_out = 2
    G_A2B=dl.MEBCRN(input_shape=(hgt,wdt,d_ech),
                    n_outputs=n_out,
                    n_res_blocks=5,
                    n_downsamplings=2,
                    filters=args.n_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

# restore
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

# ==============================================================================
# =                                 test step                                  =
# ==============================================================================

@tf.function
def sample_A2B(A):
    indx_B = tf.concat([tf.zeros_like(A[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(A[:,:,:,:2],dtype=tf.int32)],axis=-1)
    indx_PM =tf.concat([tf.zeros_like(A[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(A[:,:,:,:1],dtype=tf.int32)],axis=-1)
    # Estimate A2B
    if args.out_vars == 'WF':
        A2B_WF_abs = G_A2B(A, training=True)
        A2B_WF_abs = tf.where(A[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)
        A2B_PM = tf.zeros_like(A[:,:,:,:2])
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
    elif args.out_vars == 'PM':
        A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0.0)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,A[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
    elif args.out_vars == 'WF-PM':
        A2B_abs = G_A2B(A, training=True)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_WF_abs,A2B_PM = tf.dynamic_partition(A2B_abs,indx_B,num_partitions=2)
            A2B_WF_abs = tf.reshape(A2B_WF_abs,A[:,:,:,:2].shape)
            A2B_PM = tf.reshape(A2B_PM,A[:,:,:,4:].shape)
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,A[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)

    return A2B_abs

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

save_dir = py.join(args.experiment_dir, 'samples_testing', 'Uncertainty')
py.mkdir(save_dir)

dropout_levels = [0.01,0.02,0.05,0.1,0.2,0.5]
# dropout_Y = tf.zeros((len(dropout_levels),1,hgt,wdt,4))

i = 0
for d_lev in dropout_levels:
    print('Dropout level:',str(d_lev))
    for layer in G_A2B.layers:
        if isinstance(layer,tf.keras.layers.Dropout):
            layer.rate = d_lev

    d_lev_Y = tf.zeros_like(testX)
    sl = 0
    for X_batch in A_ds_test.as_numpy_iterator():
        X_batch = tf.expand_dims(X_batch,0)
        slice_d_lev_Y = sample_A2B(X_batch)
        slice_d_lev_Y = tf.where(X_batch[:,:,:,:4]!=0.0,slice_d_lev_Y,0.0)
        if sl == 0:
            d_lev_Y = slice_d_lev_Y
        else:
            d_lev_Y = tf.concat([d_lev_Y,slice_d_lev_Y],axis=0)
        sl += 1

    if i==0:
        dropout_Y = tf.expand_dims(d_lev_Y,axis=0)
    else:
        d_lev_Y = tf.expand_dims(d_lev_Y,axis=0)
        dropout_Y = tf.concat([dropout_Y,d_lev_Y],axis=0)
    i+=1

print('Computing Uncertainty Maps...')
uncertain_maps = tf.math.reduce_variance(dropout_Y,axis=0)

print('Saving images...')
for n_sample in range(len(samples)):
    out_Y_1 = dropout_Y[0]
    A2B_1 = out_Y_1[n_sample,:,:,:]
    out_Y_end = dropout_Y[-1]
    A2B_end = out_Y_end[n_sample,:,:,:]
    B = testY[n_sample,:,:,:]

    fig, axs = plt.subplots(figsize=(14, 6), nrows=2, ncols=4)

    # Ground-truth W/F images in the first column
    w_aux = np.squeeze(np.abs(tf.complex(B[:,:,0],B[:,:,1])))
    f_aux = np.squeeze(np.abs(tf.complex(B[:,:,2],B[:,:,3])))
    W_ok = axs[0,0].imshow(w_aux, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok, ax=axs[0,0])
    axs[0,0].axis('off')
    F_ok = axs[1,0].imshow(f_aux, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok, ax=axs[1,0])
    axs[1,0].axis('off')

    # Smallest dropout W/F images in the second column
    wn_aux_1 = A2B_1[:,:,0]
    fn_aux_1 = A2B_1[:,:,1]
    Wn_ok_1 = axs[0,1].imshow(wn_aux_1, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Wn_ok_1, ax=axs[0,1])
    axs[0,1].axis('off')
    Fn_ok_1 = axs[1,1].imshow(fn_aux_1, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Fn_ok_1, ax=axs[1,1])
    axs[1,1].axis('off')

    # Largest dropout W/F images in the third column
    wn_aux_end = A2B_end[:,:,0]
    fn_aux_end = A2B_end[:,:,1]
    Wn_ok_end = axs[0,2].imshow(wn_aux_end, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Wn_ok_end, ax=axs[0,2])
    axs[0,2].axis('off')
    Fn_ok_end = axs[1,2].imshow(fn_aux_end, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Fn_ok_end, ax=axs[1,2])
    axs[1,2].axis('off')

    # Uncertainty images in the fourth column
    wn_unc = uncertain_maps[n_sample,:,:,0]
    fn_unc = uncertain_maps[n_sample,:,:,1]
    Wn_unc = axs[0,3].imshow(wn_unc, cmap='cividis', interpolation='none', vmin=0, vmax=0.001)
    fig.colorbar(Wn_unc, ax=axs[0,3])
    axs[0,3].axis('off')
    Fn_unc = axs[1,3].imshow(fn_unc, cmap='magma', interpolation='none', vmin=0, vmax=0.001)
    fig.colorbar(Fn_unc, ax=axs[1,3])
    axs[1,3].axis('off')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0.1, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_dir+'/sample'+str(n_sample).zfill(3)+'.png',
        bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)
