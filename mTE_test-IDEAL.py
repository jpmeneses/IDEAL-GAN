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
py.arg('--dataset', type=str, default='multiTE', choices=['multiTE','phantom'])
py.arg('--n_echoes', type=int, default=6)
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
workbook = xlsxwriter.Workbook(py.join(test_args.experiment_dir, args.dataset+'-metrics.xlsx'))
ws_MAE = workbook.add_worksheet('MAE')
ws_MAE.write(0,0,'A2B Water')
ws_MAE.write(0,1,'A2B Fat')
ws_MAE.write(0,2,'A2B PDFF')
ws_MAE.write(0,3,'A2B R2*')
ws_MAE.write(0,4,'A2B FieldMap')
ws_MAE.write(0,5,'B2A2B Water')
ws_MAE.write(0,6,'B2A2B Fat')
ws_MAE.write(0,7,'B2A2B PDFF')
ws_MAE.write(0,8,'B2A2B R2*')
ws_MAE.write(0,9,'B2A2B FieldMap')

ws_MSE = workbook.add_worksheet('RMSE')
ws_MSE.write(0,0,'A2B Water')
ws_MSE.write(0,1,'A2B Fat')
ws_MSE.write(0,2,'A2B PDFF')
ws_MSE.write(0,3,'A2B R2*')
ws_MSE.write(0,4,'A2B FieldMap')
ws_MSE.write(0,5,'B2A2B Water')
ws_MSE.write(0,6,'B2A2B Fat')
ws_MSE.write(0,7,'B2A2B PDFF')
ws_MSE.write(0,8,'B2A2B R2*')
ws_MSE.write(0,9,'B2A2B FieldMap')

ws_SSIM = workbook.add_worksheet('SSIM')
ws_SSIM.write(0,0,'A2B Water')
ws_SSIM.write(0,1,'A2B Fat')
ws_SSIM.write(0,2,'A2B PDFF')
ws_SSIM.write(0,3,'A2B R2*')
ws_SSIM.write(0,4,'A2B FieldMap')
ws_SSIM.write(0,5,'B2A2B Water')
ws_SSIM.write(0,6,'B2A2B Fat')
ws_SSIM.write(0,7,'B2A2B PDFF')
ws_SSIM.write(0,8,'B2A2B R2*')
ws_SSIM.write(0,9,'B2A2B FieldMap')

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

############################################################
############### DIRECTORIES AND FILENAMES ##################
############################################################
dataset_dir = '../MRI-Datasets/'
dataset_hdf5 = 'UNet-' + args.dataset + '/' + args.dataset + '_GC_192_complex_2D.hdf5'

############################################################
###################### LOAD DATASET ########################
############################################################
f1 = h5py.File(dataset_dir + dataset_hdf5, 'r')
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

# restore
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample_A2B(A,TE):
    if args.te_input:
        A2B_PM = G_A2B([A,(TE-1e-3)/(11.5*1e-3)], training=False)
    else:
        A2B_PM = G_A2B(A, training=False)
    A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0)
    A2B_WF, A2B2A = wf.acq_to_acq(A,A2B_PM,TE)
    A2B = tf.concat([A2B_WF,A2B_PM],axis=-1)
    return A2B, A2B2A


@tf.function
def sample_B2A(B,TE):
    B2A = wf.IDEAL_model(B,args.n_echoes,te=TE)
    if args.te_input:
        B2A2B_PM = G_A2B([B2A,(TE-1e-3)/(11.5*1e-3)], training=False)
    else:
        B2A2B_PM = G_A2B(B2A, training=False)
    B2A2B_WF = wf.get_rho(B2A,B2A2B_PM,TE)
    B2A2B = tf.concat([B2A2B_WF,B2A2B_PM],axis=-1)
    B2A2B = tf.where(B!=0.0,B2A2B,0)
    return B2A, B2A2B


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', args.dataset)
py.mkdir(save_dir)
i = 0

for A, TE_smp, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    TE_smp = tf.expand_dims(TE_smp,axis=0)
    B = tf.expand_dims(B,axis=0)
    A2B, A2B2A = sample_A2B(A,TE_smp)
    B2A, B2A2B = sample_B2A(B,TE_smp)
    
    fig,axs=plt.subplots(figsize=(14, 9), nrows=3, ncols=4)

    # A2B maps in the first row
    w_aux = np.squeeze(np.abs(tf.complex(A2B[:,:,:,0],A2B[:,:,:,1])))
    W_ok = axs[0,0].imshow(w_aux, cmap='bone',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok, ax=axs[0,0])
    axs[0,0].axis('off')

    f_aux = np.squeeze(np.abs(tf.complex(A2B[:,:,:,2],A2B[:,:,:,3])))
    F_ok = axs[0,1].imshow(f_aux, cmap='pink',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok, ax=axs[0,1])
    axs[0,1].axis('off')

    r2_aux = np.squeeze(A2B[:,:,:,4])
    r2_ok = axs[0,2].imshow(r2_aux*r2_sc, cmap='copper',
                       interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_ok, ax=axs[0,2])
    axs[0,2].axis('off')

    field_aux = np.squeeze(A2B[:,:,:,5])
    field_ok = axs[0,3].imshow(field_aux*fm_sc, cmap='twilight',
                          interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_ok, ax=axs[0,3])
    axs[0,3].axis('off')

    # B2A2B maps in the second row
    w_aux_2 = np.squeeze(np.abs(tf.complex(B2A2B[:,:,:,0],B2A2B[:,:,:,1])))
    W_ok_2 = axs[1,0].imshow(w_aux_2, cmap='bone',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_ok_2, ax=axs[1,0])
    axs[1,0].axis('off')

    f_aux_2 = np.squeeze(np.abs(tf.complex(B2A2B[:,:,:,2],B2A2B[:,:,:,3])))
    F_ok_2 = axs[1,1].imshow(f_aux_2, cmap='pink',
                      interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_ok_2, ax=axs[1,1])
    axs[1,1].axis('off')

    r2_aux_2 = np.squeeze(B2A2B[:,:,:,4])
    r2_ok_2 = axs[1,2].imshow(r2_aux_2*r2_sc, cmap='copper',
                       interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_ok_2, ax=axs[1,2])
    axs[1,2].axis('off')

    field_aux_2 = np.squeeze(B2A2B[:,:,:,5])
    field_ok_2 = axs[1,3].imshow(field_aux_2*fm_sc, cmap='twilight',
                          interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_ok_2, ax=axs[1,3])
    axs[1,3].axis('off')

    # Ground-truth in the third row
    wn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,0],B[:,:,:,1])))
    W_unet = axs[2,0].imshow(wn_aux, cmap='bone',
                        interpolation='none', vmin=0, vmax=1)
    fig.colorbar(W_unet, ax=axs[2,0])
    axs[2,0].axis('off')

    fn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,2],B[:,:,:,3])))
    F_unet = axs[2,1].imshow(fn_aux, cmap='pink',
                        interpolation='none', vmin=0, vmax=1)
    fig.colorbar(F_unet, ax=axs[2,1])
    axs[2,1].axis('off')

    r2n_aux = np.squeeze(B[:,:,:,4])
    r2_unet = axs[2,2].imshow(r2n_aux*r2_sc, cmap='copper',
                         interpolation='none', vmin=0, vmax=r2_sc)
    fig.colorbar(r2_unet, ax=axs[2,2])
    axs[2,2].axis('off')

    fieldn_aux = np.squeeze(B[:,:,:,5])
    field_unet = axs[2,3].imshow(fieldn_aux*fm_sc, cmap='twilight',
                            interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    fig.colorbar(field_unet, ax=axs[2,3])
    axs[2,3].axis('off')

    fig.suptitle('TE1/dTE: '+str([TE_smp[0,0].numpy(),np.mean(np.diff(TE_smp))]), fontsize=18)

    # plt.show()
    plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
    tl.make_space_above(axs,topmargin=0.6)
    plt.savefig(save_dir+'/sample'+str(i).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
    plt.close(fig)

    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0

    PDFF_aux_2 = f_aux_2/(w_aux_2+f_aux_2)
    PDFF_aux_2[np.isnan(PDFF_aux_2)] = 0.0

    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0

    # Export to Excel file
    # MSE
    MSE_w = np.sqrt(np.mean(tf.square(w_aux-wn_aux), axis=(0,1)))
    MSE_f = np.sqrt(np.mean(tf.square(f_aux-fn_aux), axis=(0,1)))
    MSE_pdff = np.sqrt(np.mean(tf.square(PDFF_aux-PDFFn_aux), axis=(0,1)))
    MSE_r2 = np.sqrt(np.mean(tf.square(r2_aux-r2n_aux), axis=(0,1)))
    MSE_fm = np.sqrt(np.mean(tf.square(field_aux-fieldn_aux), axis=(0,1)))

    MSE_w_2 = np.sqrt(np.mean(tf.square(w_aux_2-wn_aux), axis=(0,1)))
    MSE_f_2 = np.sqrt(np.mean(tf.square(f_aux_2-fn_aux), axis=(0,1)))
    MSE_pdff_2 = np.sqrt(np.mean(tf.square(PDFF_aux_2-PDFFn_aux), axis=(0,1)))
    MSE_r2_2 = np.sqrt(np.mean(tf.square(r2_aux_2-r2n_aux), axis=(0,1)))
    MSE_fm_2 = np.sqrt(np.mean(tf.square(field_aux_2-fieldn_aux), axis=(0,1)))

    ws_MSE.write(i+1,0,MSE_w)
    ws_MSE.write(i+1,1,MSE_f)
    ws_MSE.write(i+1,2,MSE_pdff)
    ws_MSE.write(i+1,3,MSE_r2)
    ws_MSE.write(i+1,4,MSE_fm)
    ws_MSE.write(i+1,5,MSE_w)
    ws_MSE.write(i+1,6,MSE_f)
    ws_MSE.write(i+1,7,MSE_pdff)
    ws_MSE.write(i+1,8,MSE_r2)
    ws_MSE.write(i+1,9,MSE_fm)

    # MAE
    MAE_w = np.mean(tf.abs(w_aux-wn_aux), axis=(0,1))
    MAE_f = np.mean(tf.abs(f_aux-fn_aux), axis=(0,1))
    MAE_pdff = np.mean(tf.abs(PDFF_aux-PDFFn_aux), axis=(0,1))
    MAE_r2 = np.mean(tf.abs(r2_aux-r2n_aux), axis=(0,1))
    MAE_fm = np.mean(tf.abs(field_aux-fieldn_aux), axis=(0,1))

    MAE_w_2 = np.mean(tf.abs(w_aux_2-wn_aux), axis=(0,1))
    MAE_f_2 = np.mean(tf.abs(f_aux_2-fn_aux), axis=(0,1))
    MAE_pdff_2 = np.mean(tf.abs(PDFF_aux_2-PDFFn_aux), axis=(0,1))
    MAE_r2_2 = np.mean(tf.abs(r2_aux_2-r2n_aux), axis=(0,1))
    MAE_fm_2 = np.mean(tf.abs(field_aux_2-fieldn_aux), axis=(0,1))

    ws_MAE.write(i+1,0,MAE_w)
    ws_MAE.write(i+1,1,MAE_f)
    ws_MAE.write(i+1,2,MAE_pdff)
    ws_MAE.write(i+1,3,MAE_r2)
    ws_MAE.write(i+1,4,MAE_fm)
    ws_MAE.write(i+1,5,MAE_w_2)
    ws_MAE.write(i+1,6,MAE_f_2)
    ws_MAE.write(i+1,7,MAE_pdff_2)
    ws_MAE.write(i+1,8,MAE_r2_2)
    ws_MAE.write(i+1,9,MAE_fm_2)

    # SSIM
    w_ssim = structural_similarity(w_aux,wn_aux,multichannel=False)
    f_ssim = structural_similarity(f_aux,fn_aux,multichannel=False)
    pdff_ssim = structural_similarity(PDFF_aux,PDFFn_aux,multichannel=False)
    r2_ssim = structural_similarity(r2_aux,r2n_aux,multichannel=False)
    fm_ssim = structural_similarity(field_aux,fieldn_aux,multichannel=False)

    w_ssim_2 = structural_similarity(w_aux_2,wn_aux,multichannel=False)
    f_ssim_2 = structural_similarity(f_aux_2,fn_aux,multichannel=False)
    pdff_ssim_2 = structural_similarity(PDFF_aux_2,PDFFn_aux,multichannel=False)
    r2_ssim_2 = structural_similarity(r2_aux_2,r2n_aux,multichannel=False)
    fm_ssim_2 = structural_similarity(field_aux_2,fieldn_aux,multichannel=False)

    ws_SSIM.write(i+1,0,w_ssim)
    ws_SSIM.write(i+1,1,f_ssim)
    ws_SSIM.write(i+1,2,pdff_ssim)
    ws_SSIM.write(i+1,3,r2_ssim)
    ws_SSIM.write(i+1,4,fm_ssim)
    ws_SSIM.write(i+1,5,w_ssim_2)
    ws_SSIM.write(i+1,6,f_ssim_2)
    ws_SSIM.write(i+1,7,pdff_ssim_2)
    ws_SSIM.write(i+1,8,r2_ssim_2)
    ws_SSIM.write(i+1,9,fm_ssim_2)
    
    i += 1

workbook.close()