import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras_unet.models import custom_unet

import DLlib as dl
import pylib as py
import tf2lib as tl
import wflib as wf

import data

import matplotlib.pyplot as plt
import tqdm
import h5py
import xlsxwriter
from skimage.metrics import structural_similarity

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--te_input', type=bool, default=False)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                   excel                                    =
# ==============================================================================
workbook = xlsxwriter.Workbook(py.join(args.experiment_dir, 'metrics.xlsx'))
ws_MAE = workbook.add_worksheet('MAE')
ws_MAE.write(0,0,'Water')
ws_MAE.write(0,1,'Fat')
ws_MAE.write(0,2,'PDFF')
ws_MAE.write(0,3,'R2*')
ws_MAE.write(0,4,'FieldMap')

ws_MSE = workbook.add_worksheet('RMSE')
ws_MSE.write(0,0,'Water')
ws_MSE.write(0,1,'Fat')
ws_MSE.write(0,2,'PDFF')
ws_MSE.write(0,3,'R2*')
ws_MSE.write(0,4,'FieldMap')

ws_SSIM = workbook.add_worksheet('SSIM')
ws_SSIM.write(0,0,'Water')
ws_SSIM.write(0,1,'Fat')
ws_SSIM.write(0,1,'PDFF')
ws_SSIM.write(0,2,'R2*')
ws_SSIM.write(0,3,'FieldMap')

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
ech_idx = args.n_echoes * 2

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../MRI-Datasets/'
dataset_hdf5_1 = 'UNet-JGalgani/JGalgani_GC_192_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, ech_idx, complex_data=(args.G_model=='complex'))

dataset_hdf5_4 = 'UNet-Volunteers/Volunteers_GC_192_complex_2D.hdf5'
acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx, complex_data=(args.G_model=='complex'))

############################################################
################# DATASET PARTITIONS #######################
############################################################

n1_div = 248 # 65
n4_div = 434 # 113

testX   = np.concatenate((acqs_1[:n1_div,:,:,:],acqs_4[:n4_div,:,:,:]),axis=0)
testY   = np.concatenate((out_maps_1[:n1_div,:,:,:],out_maps_4[:n4_div,:,:,:]),axis=0)

# Overall dataset statistics
len_dataset,hgt,wdt,d_ech = np.shape(testX)
_,_,_,n_out = np.shape(testY)
echoes = int(d_ech/2)
r2_sc,fm_sc = 200,300

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY))
A_B_dataset_test.batch(1)

# model
with tf.device('/cpu:0'):
    if args.G_model == 'encod-decod':
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                filters=args.n_G_filters,
                                R2_self_attention=args.R2_SelfAttention,
                                FM_self_attention=args.FM_SelfAttention)
    elif args.G_model == 'U-Net':
        G_A2B = custom_unet(input_shape=(hgt,wdt,d_ech),
                            num_classes=2,
                            use_attention=args.FM_SelfAttention,
                            filters=args.n_G_filters)
    elif args.G_model == 'MEBCRN':
        G_A2B  =  dl.MEBCRN(input_shape=(hgt,wdt,d_ech),
                            n_res_blocks=5,
                            n_downsamplings=2,
                            filters=args.n_G_filters,
                            self_attention=args.FM_SelfAttention)
    else:
        raise(NameError('Unrecognized Generator Architecture'))

    # restore
    tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample_A2B(A,TE=None):
    indx_PM =tf.concat([tf.zeros_like(A[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(A[:,:,:,:1],dtype=tf.int32)],axis=-1)
    if args.te_input and (TE is not(None)):
        A2B_PM = G_A2B([A,TE], training=False)
    else:
        A2B_PM = G_A2B(A, training=False)
    if args.G_model == 'U-Net':
        orig_shape = A2B_PM.shape
        A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
        A2B_R2 = tf.reshape(A2B_R2,A[:,:,:,:1].shape)
        A2B_FM = tf.reshape(A2B_FM,A[:,:,:,:1].shape)
        A2B_FM = (A2B_FM - 0.5) * 2
        A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
    A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0)
    A2B_WF, A2B2A = wf.acq_to_acq(A,A2B_PM,TE)
    A2B = tf.concat([A2B_WF,A2B_PM],axis=-1)
    return A2B, A2B2A


@tf.function
def sample_B2A(B,TE=None):
    indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
    B2A = wf.IDEAL_model(B,echoes)
    if args.te_input and (TE is not(None)):
        B2A2B_PM = G_A2B([B2A,TE], training=False)
    else:
        B2A2B_PM = G_A2B(B2A, training=False)
    if args.G_model == 'U-Net':
        orig_shape = B2A2B_PM.shape
        B2A2B_R2, B2A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
        B2A2B_R2 = tf.reshape(B2A2B_R2,B[:,:,:,:1].shape)
        B2A2B_FM = tf.reshape(B2A2B_FM,B[:,:,:,:1].shape)
        B2A2B_FM = (B2A2B_FM - 0.5) * 2
        B2A2B_PM = tf.concat([B2A2B_R2,B2A2B_FM],axis=-1)
        B2A2B_PM = tf.reshape(B2A2B_PM,orig_shape)
    B2A2B_WF = wf.get_rho(B2A,B2A2B_PM,TE)
    B2A2B = tf.concat([B2A2B_WF,B2A2B_PM],axis=-1)
    return B2A, B2A2B


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0

if args.te_input:
    TE = wf.gen_TEvar(args.n_echoes,args.batch_size,orig=True)

for A, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    B = tf.expand_dims(B,axis=0)

    if args.te_input:
        A2B, A2B2A = sample_A2B(A,TE)
    else:
        A2B, A2B2A = sample_A2B(A)
    
    # fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8))=plt.subplots(figsize=(14, 6),
    #                                                         nrows=2, ncols=4)
    # # Ground truth in the first row
    w_aux = np.squeeze(np.abs(tf.complex(A2B[:,:,:,0],A2B[:,:,:,1])))
    # W_ok = ax1.imshow(w_aux, cmap='bone',
    #                   interpolation='none', vmin=0, vmax=1)
    # fig.colorbar(W_ok, ax=ax1)
    # ax1.axis('off')

    f_aux = np.squeeze(np.abs(tf.complex(A2B[:,:,:,2],A2B[:,:,:,3])))
    # F_ok = ax2.imshow(f_aux, cmap='pink',
    #                   interpolation='none', vmin=0, vmax=1)
    # fig.colorbar(F_ok, ax=ax2)
    # ax2.axis('off')

    r2_aux = np.squeeze(A2B[:,:,:,4])
    # r2_ok = ax3.imshow(r2_aux*r2_sc, cmap='copper',
    #                    interpolation='none', vmin=0, vmax=r2_sc)
    # fig.colorbar(r2_ok, ax=ax3)
    # ax3.axis('off')

    field_aux = np.squeeze(A2B[:,:,:,5])
    # field_ok = ax4.imshow(field_aux*fm_sc, cmap='twilight',
    #                       interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    # fig.colorbar(field_ok, ax=ax4)
    # ax4.axis('off')

    # # Computed maps in the second row
    wn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,0],B[:,:,:,1])))
    # W_unet = ax5.imshow(wn_aux, cmap='bone',
    #                     interpolation='none', vmin=0, vmax=1)
    # fig.colorbar(W_unet, ax=ax5)
    # ax5.axis('off')

    fn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,2],B[:,:,:,3])))
    # F_unet = ax6.imshow(fn_aux, cmap='pink',
    #                     interpolation='none', vmin=0, vmax=1)
    # fig.colorbar(F_unet, ax=ax6)
    # ax6.axis('off')

    r2n_aux = np.squeeze(B[:,:,:,4])
    # r2_unet = ax7.imshow(r2n_aux*r2_sc, cmap='copper',
    #                      interpolation='none', vmin=0, vmax=r2_sc)
    # fig.colorbar(r2_unet, ax=ax7)
    # ax7.axis('off')

    fieldn_aux = np.squeeze(B[:,:,:,5])
    # field_unet = ax8.imshow(fieldn_aux*fm_sc, cmap='twilight',
    #                         interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
    # fig.colorbar(field_unet, ax=ax8)
    # ax8.axis('off')

    # # plt.show()
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
    #             hspace = 0.1, wspace = 0)
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig(save_dir+'/sample'+str(i).zfill(3)+'.png',bbox_inches = 'tight',
    #     pad_inches = 0)
    # plt.close(fig)

    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0

    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0

    # Export to Excel file
    # MSE
    MSE_w = np.sqrt(np.mean(tf.square(w_aux-wn_aux), axis=(0,1)))
    MSE_f = np.sqrt(np.mean(tf.square(f_aux-fn_aux), axis=(0,1)))
    MSE_pdff = np.sqrt(np.mean(tf.square(PDFF_aux-PDFFn_aux), axis=(0,1)))
    MSE_r2 = np.sqrt(np.mean(tf.square(r2_aux-r2n_aux), axis=(0,1)))
    MSE_fm = np.sqrt(np.mean(tf.square(field_aux-fieldn_aux), axis=(0,1)))

    ws_MSE.write(i+1,0,MSE_w)
    ws_MSE.write(i+1,1,MSE_f)
    ws_MSE.write(i+1,2,MSE_pdff)
    ws_MSE.write(i+1,3,MSE_r2)
    ws_MSE.write(i+1,4,MSE_fm)

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