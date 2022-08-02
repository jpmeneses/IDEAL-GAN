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
py.arg('--out_vars', default='PM', choices=['WF','PM','WF-PM'])
py.arg('--te_input', type=bool, default=False)
py.arg('--n_filters', type=int, default=32)
py.arg('--batch_size', type=int, default=1)
py.arg('--D1_SelfAttention',type=bool, default=False)
py.arg('--D2_SelfAttention',type=bool, default=True)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                   excel                                    =
# ==============================================================================
workbook = xlsxwriter.Workbook(py.join(args.experiment_dir, 'uncertainty.xlsx'))
ws_unc = workbook.add_worksheet('Mean Unc')
ws_unc.write(0,0,'Num. Sample')
ws_unc.write(0,1,'Water')
ws_unc.write(0,2,'Fat')
ws_unc.write(0,3,'R2*')
ws_unc.write(0,4,'FieldMap')

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
samples =  [12,36,56,78,104,128,151,173,195,212,238,284,260,307,331,354,379,
            404,429,455,478,500,527,548,571,596,621,648,671]
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
if args.G_model == 'multi-decod' or args.G_model == 'encod-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,d_ech),
                                te_input=args.te_input,
                                filters=args.n_filters,
                                dropout=0.01,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                filters=args.n_filters,
                                dropout=0.01,
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
def sample_A2B(A,TE=None):
    indx_B = tf.concat([tf.zeros_like(A[:,:,:,:4],dtype=tf.int32),
                        tf.ones_like(A[:,:,:,:2],dtype=tf.int32)],axis=-1)
    indx_B_abs = tf.concat([tf.zeros_like(A[:,:,:,:2],dtype=tf.int32),
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
        if args.te_input:
            A2B_PM = G_A2B([A,TE], training=True)
        else:
            A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0.0)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,A[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
    elif args.out_vars == 'WF-PM':
        if args.te_input:
            A2B_abs = G_A2B([A,TE], training=True)
        else:
            A2B_abs = G_A2B(A, training=True)
        A2B_abs = tf.where(A[:,:,:,:4]!=0.0,A2B_abs,0.0)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_WF_abs,A2B_PM = tf.dynamic_partition(A2B_abs,indx_B_abs,num_partitions=2)
            A2B_WF_abs = tf.reshape(A2B_WF_abs,A[:,:,:,:2].shape)
            A2B_PM = tf.reshape(A2B_PM,A[:,:,:,:2].shape)
            A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
            A2B_R2 = tf.reshape(A2B_R2,A[:,:,:,:1].shape)
            A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)

    return A2B_abs

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

save_dir = py.join(args.experiment_dir, 'samples_testing', 'Uncertainty')
py.mkdir(save_dir)

dropout_levels = [0.01,0.02,0.05,0.1,0.2,0.5]

if args.te_input:
    TE_smp = wf.gen_TEvar(args.n_echoes,args.batch_size,orig=True)
else:
    TE_smp = None

i = 0
for d_lev in dropout_levels:
    print('Dropout level:',str(d_lev))
    for layer in G_A2B.layers:
        if isinstance(layer,tf.keras.layers.Dropout):
            layer.rate = d_lev

    sl = 0
    for X_batch in A_ds_test.as_numpy_iterator():
        X_batch = tf.expand_dims(X_batch,0)
        slice_d_lev_Y = sample_A2B(X_batch,TE=TE_smp)
        slice_d_lev_Y = tf.where(X_batch[:,:,:,:4]!=0.0,slice_d_lev_Y,0.0)
        # PDFF_batch = X_batch[:,:,:,1]/(X_batch[:,:,:,0]+X_batch[:,:,:,1])
        # PDFF_batch = tf.where(tf.math.is_nan(PDFF_batch),0.0,PDFF_batch)
        # PDFF_batch = tf.expand_dims(PDFF_batch,axis=-1)
        # slice_d_lev_Y = tf.concat([slice_d_lev_Y,PDFF_batch],axis=-1)
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
# uncertain_maps = tf.where(tf.math.is_nan(uncertain_maps),0.0,uncertain_maps)

print('Saving images...')
for n_sample in range(len_dataset):
    wn_unc = uncertain_maps[n_sample,:,:,0]
    fn_unc = uncertain_maps[n_sample,:,:,1]
    r2n_unc = uncertain_maps[n_sample,:,:,2]*np.square(r2_sc)
    FMn_unc = uncertain_maps[n_sample,:,:,3]*np.square(fm_sc)

    out_Y_1 = dropout_Y[0]
    A2B_1 = out_Y_1[n_sample,:,:,:]
    out_Y_mid = dropout_Y[len(dropout_levels)//2]
    A2B_mid = out_Y_mid[n_sample,:,:,:]
    out_Y_end = dropout_Y[-1]
    A2B_end = out_Y_end[n_sample,:,:,:]
    B = testY[n_sample,:,:,:]

    # ==========================================================================
    # =                           Fig 1: W/F images                            =
    # ==========================================================================
    fig, axs = plt.subplots(figsize=(14, 6), nrows=2, ncols=4)

    # Smallest dropout W/F images in the first column
    wn_aux_1 = A2B_1[:,:,0]
    fn_aux_1 = A2B_1[:,:,1]
    Wn_ok_1 = axs[0,0].imshow(wn_aux_1, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Wn_ok_1, ax=axs[0,0])
    axs[0,0].axis('off')
    Fn_ok_1 = axs[1,0].imshow(fn_aux_1, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Fn_ok_1, ax=axs[1,0])
    axs[1,0].axis('off')

    # Medium dropout W/F images in the second column
    wn_aux_mid = A2B_mid[:,:,0]
    fn_aux_mid = A2B_mid[:,:,1]
    Wn_ok_mid = axs[0,1].imshow(wn_aux_mid, cmap='bone', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Wn_ok_mid, ax=axs[0,1])
    axs[0,1].axis('off')
    Fn_ok_mid = axs[1,1].imshow(fn_aux_mid, cmap='pink', interpolation='none', vmin=0, vmax=1)
    fig.colorbar(Fn_ok_mid, ax=axs[1,1])
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
    plt.savefig(save_dir+'/WF_sample'+str(samples[n_sample]).zfill(3)+'.png',
        bbox_inches = 'tight', pad_inches = 0)
    plt.close(fig)

    # ==========================================================================
    # =                       Fig 2: Quantitative maps                         =
    # ==========================================================================
    if not(args.out_vars == 'WF'):
        fig, axs = plt.subplots(figsize=(14, 6), nrows=2, ncols=4)

        # Smallest dropout param maps in the first column
        r2n_aux_1 = A2B_1[:,:,2]*r2_sc
        FMn_aux_1 = A2B_1[:,:,3]*fm_sc
        r2n_ok_1 = axs[0,0].imshow(r2n_aux_1, cmap='copper', interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2n_ok_1, ax=axs[0,0])
        axs[0,0].axis('off')
        FMn_ok_1 = axs[1,0].imshow(FMn_aux_1, cmap='twilight', interpolation='none', vmin=-fm_sc, vmax=fm_sc)
        fig.colorbar(FMn_ok_1, ax=axs[1,0])
        axs[1,0].axis('off')

        # Medium dropout param maps in the second column
        r2n_aux_mid = A2B_mid[:,:,2]*r2_sc
        FMn_aux_mid = A2B_mid[:,:,3]*fm_sc
        r2n_ok_mid = axs[0,1].imshow(r2n_aux_mid, cmap='copper', interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2n_ok_mid, ax=axs[0,1])
        axs[0,1].axis('off')
        FMn_ok_mid = axs[1,1].imshow(FMn_aux_mid, cmap='twilight', interpolation='none', vmin=-fm_sc, vmax=fm_sc)
        fig.colorbar(FMn_ok_mid, ax=axs[1,1])
        axs[1,1].axis('off')

        # Largest dropout param maps in the third column
        r2n_aux_end = A2B_end[:,:,2]*r2_sc
        FMn_aux_end = A2B_end[:,:,3]*fm_sc
        r2n_ok_end = axs[0,2].imshow(r2n_aux_end, cmap='copper', interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2n_ok_end, ax=axs[0,2])
        axs[0,2].axis('off')
        FMn_ok_end = axs[1,2].imshow(FMn_aux_end, cmap='twilight', interpolation='none', vmin=-fm_sc, vmax=fm_sc)
        fig.colorbar(FMn_ok_end, ax=axs[1,2])
        axs[1,2].axis('off')

        # Uncertainty images in the fourth column
        r2n_ok_unc = axs[0,3].imshow(r2n_unc, cmap='plasma', interpolation='none', vmin=0, vmax=10.0)
        fig.colorbar(r2n_ok_unc, ax=axs[0,3])
        axs[0,3].axis('off')
        FMn_ok_unc = axs[1,3].imshow(FMn_unc, cmap='viridis', interpolation='none', vmin=0, vmax=10.0)
        fig.colorbar(FMn_ok_unc, ax=axs[1,3])
        axs[1,3].axis('off')

        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0.1, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_dir+'/PM_sample'+str(samples[n_sample]).zfill(3)+'.png',
            bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)

    # ==========================================================================
    # =                       Save mean uncertainties                          =
    # ==========================================================================
    unc_w = np.mean(wn_unc)
    unc_f = np.mean(fn_unc)
    unc_r2 = np.mean(r2n_unc)
    unc_fm = np.mean(FMn_unc)

    ws_unc.write(n_sample+1,0,n_sample)
    ws_unc.write(n_sample+1,1,unc_w)
    ws_unc.write(n_sample+1,2,unc_f)
    ws_unc.write(n_sample+1,3,unc_r2)
    ws_unc.write(n_sample+1,4,unc_fm)

workbook.close()