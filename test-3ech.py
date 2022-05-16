import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import pylib as py
import tf2lib as tl
import wflib as wf

import data
import module

import matplotlib.pyplot as plt
import tqdm
import h5py
import xlsxwriter
from skimage.metrics import structural_similarity

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--G_model', default='encod-decod', choices=['encod-decod','U-Net','MEBCRN'])
py.arg('--te_input', type=bool, default=True)
py.arg('--n_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=1)
py.arg('--R2_SelfAttention',type=bool, default=False)
py.arg('--FM_SelfAttention',type=bool, default=True)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# ==============================================================================
# =                                   excel                                    =
# ==============================================================================
workbook = xlsxwriter.Workbook(py.join(test_args.experiment_dir, '3ech-metrics.xlsx'))
ws_MAE = workbook.add_worksheet('MAE')
ws_MAE.write(0,0,'Water')
ws_MAE.write(0,1,'Fat')
ws_MAE.write(0,2,'PDFF')
ws_MAE.write(0,3,'R2*')
ws_MAE.write(0,4,'FieldMap')

ws_RMSE = workbook.add_worksheet('RMSE')
ws_RMSE.write(0,0,'Water')
ws_RMSE.write(0,1,'Fat')
ws_RMSE.write(0,2,'PDFF')
ws_RMSE.write(0,3,'R2*')
ws_RMSE.write(0,4,'FieldMap')

ws_SSIM = workbook.add_worksheet('SSIM')
ws_SSIM.write(0,0,'Water')
ws_SSIM.write(0,1,'Fat')
ws_SSIM.write(0,2,'PDFF')
ws_SSIM.write(0,3,'R2*')
ws_SSIM.write(0,4,'FieldMap')

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

############################################################
############### DIRECTORIES AND FILENAMES ##################
############################################################
dataset_dir = '../MRI-Datasets/'
dataset_hdf5_1 = 'UNet-multiTE/3ech_GC_192_complex_2D.hdf5'

############################################################
################### LOAD DATASET 1 #########################
############################################################
f1 = h5py.File(dataset_dir + dataset_hdf5_1, 'r')
acqs_1 = f1['Acquisitions'][...]
TEs_1 = f1['TEs'][...]
out_maps_1 = f1['OutMaps'][...]
f1.close()

idxs_list_1 = []
for nd in range(len(acqs_1)):
  if np.sum(acqs_1[nd,:,:,1])!=0.0:
    idxs_list_1.append(nd)

acqs_1 = acqs_1[idxs_list_1,:,:,:ech_idx]
TEs_1 = TEs_1[idxs_list_1,:args.n_echoes]
out_maps_1 = out_maps_1[idxs_list_1,:,:,:]

print('Num. Elements- DS1:', len(acqs_1))

############################################################
################# DATASET PARTITIONS #######################
############################################################

testX   = acqs_1
testY   = out_maps_1
TEs     = TEs_1

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

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,TEs,testY))
A_B_dataset_test.batch(1)

# model
G_A2B = module.PM_Generator(input_shape=(hgt,wdt,d_ech),
                            filters=args.n_filters,
                            te_input=args.te_input,
                            te_shape=(args.n_echoes,),
                            R2_self_attention=args.R2_SelfAttention,
                            FM_self_attention=args.FM_SelfAttention)

# model
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample_A2B(A,TE):
    if args.te_input:
        A2B_PM = G_A2B([A,(TE-1e-3)/(11.5*1e-3)], training=False)
    else:
        A2B_PM = G_A2B(A, training=False)
    A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0)
    A2B_WF, A2B2A = wf.acq_to_acq(A,A2B_PM)
    A2B = tf.concat([A2B_WF,A2B_PM],axis=-1)
    return A2B, A2B2A

# run
save_dir = py.join(args.experiment_dir, 'samples_testing', '3ech')
py.mkdir(save_dir)
i = 0

for A, TE_smp, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    TE_smp = tf.expand_dims(TE_smp,axis=0)
    B = tf.expand_dims(B,axis=0)
    A2B, A2B2A = sample_A2B(A,TE_smp)

    fig, axs = plt.subplots(figsize=(14, 6), nrows=2, ncols=4)
    
    # Ground truth in the first row
    w_aux = np.squeeze(A2B[:,:,:,0])
    W_ok = axs[0,0].imshow(w_aux, cmap='bone',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok, ax=axs[0,0])
    axs[0,0].axis('off')

    f_aux = np.squeeze(A2B[:,:,:,1])
    F_ok = axs[0,1].imshow(f_aux, cmap='pink',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok, ax=axs[0,1])
    axs[0,1].axis('off')

    r2_aux = np.squeeze(A2B[:,:,:,2])
    r2_ok = axs[0,2].imshow(r2_aux*r2_sc, cmap='copper',
                       interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_ok, ax=axs[0,2])
    axs[0,2].axis('off')

    field_aux = np.squeeze(A2B[:,:,:,3])
    field_ok = axs[0,3].imshow(field_aux*fm_sc, cmap='twilight',
                          interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_ok, ax=axs[0,3])
    axs[0,3].axis('off')

    # Computed maps in the second row
    wn_aux = np.squeeze(B[:,:,:,0])
    W_unet = axs[1,0].imshow(wn_aux, cmap='bone',
                        interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_unet, ax=axs[1,0])
    axs[1,0].axis('off')

    fn_aux = np.squeeze(B[:,:,:,1])
    F_unet = axs[1,1].imshow(fn_aux, cmap='pink',
                        interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_unet, ax=axs[1,1])
    axs[1,1].axis('off')

    r2n_aux = np.squeeze(B[:,:,:,2])
    r2_unet = axs[1,2].imshow(r2n_aux*r2_sc, cmap='copper',
                         interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_unet, ax=axs[1,2])
    axs[1,2].axis('off')

    fieldn_aux = np.squeeze(B[:,:,:,3])
    field_unet = axs[1,3].imshow(fieldn_aux*fm_sc, cmap='twilight',
                            interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_unet, ax=axs[1,3])
    axs[1,3].axis('off')

    fig.suptitle('TE1/dTE: '+str([TE_smp[0,0].numpy(),np.mean(np.diff(TE_smp))]), fontsize=18)

    # plt.show()
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.6)
    plt.savefig(save_dir+'/sample'+str(i).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)

    # PDFF measurements
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0

    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0

    # Export to Excel file
    # MSE
    RMSE_w = np.sqrt(np.mean(tf.square(w_aux-wn_aux), axis=(0,1)))
    RMSE_f = np.sqrt(np.mean(tf.square(f_aux-fn_aux), axis=(0,1)))
    RMSE_pdff = np.sqrt(np.mean(tf.square(PDFF_aux-PDFFn_aux), axis=(0,1)))
    RMSE_r2 = np.sqrt(np.mean(tf.square(r2_aux-r2n_aux), axis=(0,1)))
    RMSE_fm = np.sqrt(np.mean(tf.square(field_aux-fieldn_aux), axis=(0,1)))

    ws_RMSE.write(i+1,0,RMSE_w)
    ws_RMSE.write(i+1,1,RMSE_f)
    ws_RMSE.write(i+1,2,RMSE_pdff)
    ws_RMSE.write(i+1,3,RMSE_r2)
    ws_RMSE.write(i+1,4,RMSE_fm)

    # MAE
    MAE_w = np.mean(tf.abs(w_aux-wn_aux), axis=(0,1))
    MAE_f = np.mean(tf.abs(f_aux-fn_aux), axis=(0,1))
    MAE_pdff = np.mean(tf.abs(PDFF_aux-PDFFn_aux), axis=(0,1))
    MAE_r2 = np.mean(tf.abs(r2_aux-r2n_aux), axis=(0,1))
    MAE_fm = np.mean(tf.abs(field_aux-fieldn_aux), axis=(0,1))

    ws_MAE.write(i+1,0,MAE_w)
    ws_MAE.write(i+1,1,MAE_f)
    ws_MAE.write(i+1,2,MAE_pdff)
    ws_MAE.write(i+1,3,MAE_r2)
    ws_MAE.write(i+1,4,MAE_fm)

    # SSIM
    w_ssim = structural_similarity(w_aux,wn_aux,multichannel=False)
    f_ssim = structural_similarity(f_aux,fn_aux,multichannel=False)
    pdff_ssim = structural_similarity(PDFF_aux,PDFFn_aux,multichannel=False)
    r2_ssim = structural_similarity(r2_aux,r2n_aux,multichannel=False)
    fm_ssim = structural_similarity(field_aux,fieldn_aux,multichannel=False)

    ws_SSIM.write(i+1,0,w_ssim)
    ws_SSIM.write(i+1,1,f_ssim)
    ws_SSIM.write(i+1,2,pdff_ssim)
    ws_SSIM.write(i+1,3,r2_ssim)
    ws_SSIM.write(i+1,4,fm_ssim)
    
    i += 1

workbook.close()