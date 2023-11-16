import numpy as np
import tensorflow as tf

import tf2lib as tl
import DLlib as dl
import pylib as py
import wflib as wf
import data

import matplotlib.pyplot as plt
import tqdm
import xlsxwriter
from skimage.metrics import structural_similarity

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--dataset', type=str, default='multiTE', choices=['multiTE','3ech','JGalgani','phantom_1p5','phantom_3p0'])
py.arg('--data_size', type=int, default=192, choices=[192,384])
py.arg('--te_input', type=bool, default=True)
py.arg('--ME_layer', type=bool, default=False)
py.arg('--UQ', type=bool, default=False)
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
py.arg('--n_plot',type=int,default=50)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

if not(hasattr(args,'field')):
    py.arg('--field', type=float, default=1.5)
    ds_args = py.args()
    args.__dict__.update(ds_args.__dict__)


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

ws_MSE = workbook.add_worksheet('RMSE')
ws_MSE.write(0,0,'A2B Water')
ws_MSE.write(0,1,'A2B Fat')
ws_MSE.write(0,2,'A2B PDFF')
ws_MSE.write(0,3,'A2B R2*')
ws_MSE.write(0,4,'A2B FieldMap')
ws_MSE.write(0,5,'TE1')
ws_MSE.write(0,6,'dTE')

ws_SSIM = workbook.add_worksheet('SSIM')
ws_SSIM.write(0,0,'A2B Water')
ws_SSIM.write(0,1,'A2B Fat')
ws_SSIM.write(0,2,'A2B PDFF')
ws_SSIM.write(0,3,'A2B R2*')
ws_SSIM.write(0,4,'A2B FieldMap')

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
if args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
    dataset_hdf5 = args.dataset + '_GC_192_128_complex_2D.hdf5'
else:
    dataset_hdf5 = args.dataset + '_GC_' + str(args.data_size) + '_complex_2D.hdf5'

if args.dataset == 'JGalgani':
    num_slice_list = [21,20,24,24,21,24,21,21,24,21,21,22,27,23,22,20,24,21,21,22,20]
    rnc = True
else:
    num_slice_list = None
    rnc = False

if args.dataset == 'JGalgani' or args.dataset == '3ech':
    testX, testY=data.load_hdf5(dataset_dir,dataset_hdf5,ech_idx,num_slice_list=num_slice_list,remove_non_central=rnc,
                                acqs_data=True,te_data=False,remove_zeros=True,MEBCRN=True)
    TEs = np.ones((testX.shape[0],1),dtype=np.float32)
else:
    testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, acqs_data=True, 
                                        te_data=True,remove_zeros=not(args.dataset=='multiTE'),MEBCRN=True)
if args.dataset == 'multiTE':
    testX, testY, TEs = data.group_TEs(testX,testY,TEs,TE1=args.TE1,dTE=args.dTE,MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,n_out,hgt,wdt,n_ch = np.shape(testY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',args.n_echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)
print('TEs shape:',TEs.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,TEs,testY))
A_B_dataset_test.batch(1)

# model
if args.ME_layer:
    input_shape = (args.n_echoes,hgt,wdt,n_ch)
else:
    input_shape = (hgt,wdt,ech_idx)

if args.G_model == 'multi-decod' or args.G_model == 'encod-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=input_shape,
                                te_input=args.te_input,
                                filters=args.n_G_filters,
                                dropout=0.0,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=input_shape,
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                ME_layer=args.ME_layer,
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        n_out = 4
    elif args.out_vars == 'FM':
        n_out = 1
    else:
        n_out = 2
    G_A2B = dl.UNet(input_shape=(hgt,wdt,ech_idx),
                    n_out=n_out,
                    bayesian=args.UQ,
                    te_input=args.te_input,
                    te_shape=(args.n_echoes,),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

elif args.G_model == 'MEBCRN':
    if args.out_vars == 'WFc':
        n_out = 4
        out_activ = None
    else:
        n_out = 2
        out_activ = 'sigmoid'
    G_A2B=dl.MEBCRN(input_shape=(args.n_echoes,hgt,wdt,2),
                    n_outputs=n_out,
                    output_activation=out_activ,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

# restore
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample(A, B, TE=None):
    # Back-up A
    A_ME = A
    if not(args.ME_layer):
        A = data.A_from_MEBCRN(A)
    # Split B
    B_WF = B[:,:2,:,:,:]
    B_PM = B[:,2:,:,:,:]
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))
    # Estimate A2B
    if args.out_vars == 'WF':
        if args.te_input:
            A2B_WF = G_A2B([A,TE], training=True)
        else:
            A2B_WF = G_A2B(A, training=True)
        A2B_WF = data.B_to_MEBCRN(A2B_WF,mode='WF')
        A2B_WF = tf.where(B_WF!=0.0,A2B_WF,0.0)
        A2B_PM = tf.zeros_like(B_PM)
        A2B_R2 = A2B_PM[:,:,:,:,1:]
        A2B_FM = A2B_PM[:,:,:,:,:1]
        A2B = tf.concat([A2B_WF,A2B_PM],axis=1)
        A2B_var = None
    # elif args.out_vars == 'WFc':
    #     if args.te_input:
    #         A2B_WF = G_A2B([A,te], training=True)
    #     else:
    #         A2B_WF = G_A2B(A, training=True)
    #     A2B_WF = tf.where(B[:,:,:,:4]!=0.0,A2B_WF,0.0)
    #     # Magnitude of water/fat images
    #     A2B_WF_real = A2B_WF[:,:,:,0::2]
    #     A2B_WF_imag = A2B_WF[:,:,:,1::2]
    #     A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
    #     # Compute zero-valued param maps
    #     A2B_PM = tf.zeros_like(A2B_WF_abs)
    #     A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
    #     A2B_var = None
    elif args.out_vars == 'PM':
        if args.te_input:
            A2B_PM = G_A2B([A,TE], training=False)
        else:
            A2B_PM = G_A2B(A, training=False)
        if not(args.ME_layer):
            A2B_PM = data.B_to_MEBCRN(A2B_PM,mode='PM')
        A2B_PM = tf.where(A_ME[:,:1,:,:,:]!=0.0,A2B_PM,0.0)
        A2B_R2 = A2B_PM[:,:,:,:,1:]
        A2B_FM = A2B_PM[:,:,:,:,:1]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(A_ME[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=1)
        A2B_WF = wf.get_rho(A_ME, A2B_PM, field=args.field, te=TE)
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))
        A2B = tf.concat([A2B_WF,A2B_PM],axis=1)
        A2B_var = None
    elif args.out_vars == 'WF-PM':
        if args.te_input:
            A2B_abs = G_A2B([A,TE], training=True)
        else:
            A2B_abs = G_A2B(A, training=True)
        A2B = data.B_to_MEBCRN(A2B_abs,mode='WF-PM')
        A2B = tf.where(A_ME[:,:3,:,:,:]!=0.0,A2B,0.0)
        A2B_WF = A2B[:,:2,:,:,:]
        A2B_PM = A2B[:,2:,:,:,:]
        A2B_R2 = A2B_PM[:,:,:,:,1:]
        A2B_FM = A2B_PM[:,:,:,:,:1]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
          A2B_FM = (A2B_FM - 0.5) * 2
          A2B_FM = tf.where(A_ME[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)
          A2B_PM = tf.concat([A2B_FM,A2B_R2],axis=-1)
          A2B = tf.concat([A2B_WF,A2B_PM],axis=1)
        A2B_var = None
    # elif args.out_vars == 'FM':
    #     if args.UQ:
    #         _, A2B_FM, A2B_var = G_A2B(A, training=False)
    #     else:
    #         A2B_FM = G_A2B(A, training=False)
    #         A2B_var = None
        
    #     # A2B Masks
    #     A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
    #     if args.UQ:
    #         A2B_var = tf.where(A[:,:,:,:1]!=0.0,A2B_var,0.0)

    #     # Build A2B_PM array with zero-valued R2*
    #     A2B_PM = tf.concat([tf.zeros_like(A2B_FM),A2B_FM], axis=-1)
    #     if args.fat_char:
    #         A2B_P, A2B2A = fa.acq_to_acq(A,A2B_PM,complex_data=(args.G_model=='complex'))
    #         A2B_WF = A2B_P[:,:,:,0:4]
    #     else:
    #         A2B_WF, A2B2A = wf.acq_to_acq(A,A2B_PM,te=TE,complex_data=(args.G_model=='complex'))
    #     A2B = tf.concat([A2B_WF,A2B_PM],axis=-1)

    #     # Magnitude of water/fat images
    #     A2B_WF_real = A2B_WF[:,:,:,0::2]
    #     A2B_WF_imag = A2B_WF[:,:,:,1::2]
    #     A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
    #     A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)

    return A2B, A2B_var


# run
if args.dataset == 'multiTE':
    save_dir = py.join(args.experiment_dir, 'samples_testing', args.dataset
                        + str(int(np.round(args.TE1*1e4))) + '_' + str(int(np.round(args.dTE*1e4))))
else:
    save_dir = py.join(args.experiment_dir, 'samples_testing', args.dataset)
py.mkdir(save_dir)
i = 0

if args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
    plot_hgt = 6
else:
    plot_hgt = 9

for A, TE_smp, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    TE_smp = tf.expand_dims(TE_smp,axis=0)
    B = tf.expand_dims(B,axis=0)
    B_WF = B[:,:2,:,:,:]
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))
    
    if args.te_input:
        A2B, A2B_var = sample(A, B, TE_smp)
    else:
        A2B, A2B_var = sample(A, B)
    A2B_WF = A2B[:,:2,:,:,:]
    A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))

    w_aux = np.squeeze(A2B_WF_abs[:,0,:,:,:])
    f_aux = np.squeeze(A2B_WF_abs[:,1,:,:,:])
    if not(args.UQ):
        r2_aux = np.squeeze(A2B[:,2,:,:,1])*r2_sc
        lmax = r2_sc
        cmap = 'copper'
    else:
        r2_aux = np.squeeze(A2B_var)*fm_sc
        lmax = 30
        cmap = 'gnuplot2'
    field_aux = np.squeeze(A2B[:,2,:,:,0])
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0

    wn_aux = np.squeeze(B_WF_abs[:,0,:,:,:])
    fn_aux = np.squeeze(B_WF_abs[:,1,:,:,:])
    r2n_aux = np.squeeze(B[:,2,:,:,1])
    fieldn_aux = np.squeeze(B[:,2,:,:,0])
    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0
    
    if i%args.n_plot == 0 or args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
        fig,axs=plt.subplots(figsize=(16, plot_hgt), nrows=2, ncols=3)

        # A2B maps in the first row
        F_ok = axs[0,0].imshow(PDFF_aux, cmap='jet',
                          interpolation='none', vmin=0, vmax=1)
        fig.colorbar(F_ok, ax=axs[0,0]).ax.tick_params(labelsize=14)
        axs[0,0].axis('off')

        r2_ok = axs[0,1].imshow(r2_aux, cmap=cmap,
                                interpolation='none', vmin=0, vmax=lmax)
        fig.colorbar(r2_ok, ax=axs[0,1]).ax.tick_params(labelsize=14)
        axs[0,1].axis('off')

        field_ok = axs[0,2].imshow(field_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
        fig.colorbar(field_ok, ax=axs[0,2]).ax.tick_params(labelsize=14)
        axs[0,2].axis('off')

        # Ground-truth in the third row
        F_unet = axs[1,0].imshow(PDFFn_aux, cmap='jet',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(F_unet, ax=axs[1,0]).ax.tick_params(labelsize=14)
        axs[1,0].axis('off')

        r2_unet=axs[1,1].imshow(r2n_aux*r2_sc, cmap='copper',
                                interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2_unet, ax=axs[1,1]).ax.tick_params(labelsize=14)
        axs[1,1].axis('off')

        field_unet =axs[1,2].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
        fig.colorbar(field_unet, ax=axs[1,2]).ax.tick_params(labelsize=14)
        axs[1,2].axis('off')

        if args.te_input:
            fig.suptitle('TE1/dTE: '+str([TE_smp[0,0,0].numpy(),np.mean(np.diff(TE_smp,axis=1))]), fontsize=18)
        else:
            fig.suptitle('TE1/dTE: '+str([args.TE1,args.dTE]), fontsize=18)

        # plt.show()
        plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
        tl.make_space_above(axs,topmargin=0.6)
        plt.savefig(save_dir+'/sample'+str(i).zfill(3)+'.png',bbox_inches='tight',pad_inches=0)
        fig.clear()
        plt.close(fig)

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
    
    if args.te_input:
        ws_MSE.write(i+1,5,TE_smp[0,0,0].numpy())
        ws_MSE.write(i+1,6,np.mean(np.diff(TE_smp,axis=1)))
    else:
        ws_MSE.write(i+1,5,args.TE1)
        ws_MSE.write(i+1,6,args.dTE)

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