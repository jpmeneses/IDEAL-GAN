import numpy as np
import tensorflow as tf
import tqdm
import xlsxwriter
import matplotlib.pyplot as plt

import tf2lib as tl
import DLlib as dl
import wflib as wf
import pylib as py
import data

from matplotlib.colors import LogNorm
from skimage.metrics import structural_similarity

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--te_input', type=bool, default=False)
py.arg('--D1_SelfAttention',type=bool, default=True)
py.arg('--D2_SelfAttention',type=bool, default=False)
py.arg('--n_plot', type=int, default=30)
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

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'JGalgani_GC_192_complex_2D.hdf5'
dataset_hdf5_2 = 'INTA_GC_192_complex_2D.hdf5'
dataset_hdf5_3 = 'INTArest_GC_192_complex_2D.hdf5'
dataset_hdf5_4 = 'Volunteers_GC_192_complex_2D.hdf5'
dataset_hdf5_5 = 'Attilio_GC_192_complex_2D.hdf5'

if args.k_fold == 1:
    acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                end=320, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    testX = np.concatenate((acqs_1,acqs_2),axis=0)
    testY = np.concatenate((out_maps_1,out_maps_2),axis=0)

elif args.k_fold == 2:
    acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                start=320, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx,
                                end=798, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    testX = np.concatenate((acqs_2,acqs_3),axis=0)
    testY = np.concatenate((out_maps_2,out_maps_3),axis=0)

elif args.k_fold == 3:
    acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, ech_idx,
                                start=798, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                end=310, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    testX = np.concatenate((acqs_3,acqs_4),axis=0)
    testY = np.concatenate((out_maps_3,out_maps_4),axis=0)

elif args.k_fold == 4:
    testX, testY = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                start=310, end=1172, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))

elif args.k_fold == 5:
    acqs_4, out_maps_4 = data.load_hdf5(dataset_dir,dataset_hdf5_4, ech_idx,
                                start=1172, acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    acqs_5, out_maps_5 = data.load_hdf5(dataset_dir,dataset_hdf5_5, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True,
                                complex_data=(args.G_model=='complex'))
    testX = np.concatenate((acqs_4,acqs_5),axis=0)
    testY = np.concatenate((out_maps_4,out_maps_5),axis=0)

############################################################
################# DATASET PARTITIONS #######################
############################################################

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(testX)
_,n_out,_,_,_ = np.shape(testY)

print('Length dataset:', len_dataset)
print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',ne)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY))
A_B_dataset_test.batch(1)

# model
# model
if args.G_model == 'multi-decod' or args.G_model == 'encod-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,d_ech),
                                te_input=args.te_input,
                                filters=args.n_G_filters,
                                dropout=0.0,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(ne,hgt,wdt,n_ch),
                                te_input=args.te_input,
                                te_shape=(args.n_echoes,),
                                filters=args.n_G_filters,
                                dropout=0.0,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        n_out = 4
    elif args.out_vars == 'FM' or args.out_vars == 'R2s':
        n_out = 1
    else:
        n_out = 2
    G_A2B = dl.UNet(input_shape=(ne,hgt,wdt,n_ch),
                    n_out=n_out,
                    bayesian=args.UQ,
                    ME_layer=args.ME_layer,
                    te_input=args.te_input,
                    te_shape=(args.n_echoes,),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
    if args.out_vars == 'R2s':
        G_A2R2= dl.UNet(input_shape=(ne,hgt,wdt,1),
                        bayesian=args.UQ,
                        te_input=args.te_input,
                        te_shape=(args.n_echoes,),
                        filters=args.n_G_filters,
                        output_activation='sigmoid',
                        self_attention=args.D2_SelfAttention)

elif args.G_model == 'MEBCRN':
    if args.out_vars == 'WF-PM':
        n_out = 4
    else:
        n_out = 2
    G_A2B=dl.MEBCRN(input_shape=(hgt,wdt,d_ech),
                    n_outputs=n_out,
                    n_res_blocks=5,
                    n_downsamplings=2,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

# restore
if args.out_vars == 'R2s':
    tl.Checkpoint(dict(G_A2B=G_A2B,G_A2R2=G_A2R2), py.join(args.experiment_dir, 'checkpoints')).restore()
else:
    tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample(A, B, TE=None):
    # Estimate A2B
    if args.out_vars == 'WF':
        if args.te_input:
            A2B_WF_abs = G_A2B([A,TE], training=True)
        else:
            A2B_WF_abs = G_A2B(A, training=True)
        A2B_WF_abs = tf.where(A[:,:2,:,:,:]!=0.0,A2B_WF_abs,0.0)
        A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
        A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
        A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        A2B_var = None
    elif args.out_vars == 'PM':
        if args.te_input:
            A2B_PM = G_A2B([A,TE], training=True)
        else:
            A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(B[:,2:,:,:,:]!=0.0,A2B_PM,0.0)
        A2B_FM = A2B_PM[:,:,:,:,:1]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B[:,2:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_FM,A2B_PM[:,:,:,:,1:]],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        A2B_var = None
    elif args.out_vars == 'WF-PM':
        B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
        if args.te_input:
            A2B_abs = G_A2B([A,TE], training=True)
        else:
            A2B_abs = G_A2B(A, training=True)
        A2B_abs = tf.where(B_abs!=0.0,A2B_abs,0.0)
        A2B_WF_abs,A2B_PM = tf.dynamic_partition(A2B_abs,indx_B_abs,num_partitions=2)
        A2B_WF_abs = tf.reshape(A2B_WF_abs,B[:,:,:,:2].shape)
        A2B_PM = tf.reshape(A2B_PM,B[:,:,:,4:].shape)
        A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
        A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
        A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)
        A2B_var = None
    elif args.out_vars == 'FM':
        if args.UQ:
            _, A2B_FM, A2B_FM_var = G_A2B(A, training=False)
        else:
            A2B_FM = G_A2B(A, training=False)
            A2B_FM_var = None
        # A2B Masks
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)

        # Build A2B_PM array with zero-valued R2*
        A2B_PM = tf.concat([A2B_FM,tf.zeros_like(A2B_FM)], axis=-1)
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B = tf.concat([A2B_WF,A2B_PM],axis=1)

        # Variance map mask and attach to recon-A
        if args.UQ:
            A2B_FM_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM_var,0.0)
            A2B_R2_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_var,0.0)
            A2B_PM_var = tf.concat([A2B_FM_var,A2B_R2_var], axis=-1)
            A2B_WF_var = wf.PDFF_uncertainty(A,A2B_PM,A2B_PM_var)
            A2B_var = tf.concat([A2B_WF_var,A2B_PM_var],axis=1)

    elif args.out_vars == 'R2s':
        if not(args.G_model == 'complex'):
            A_real = A[:,:,:,0::2]
            A_imag = A[:,:,:,1::2]
            A_abs = tf.abs(tf.complex(A_real,A_imag))
        
        # Compute R2s maps using only-mag images
        if args.UQ:
            _, A2B_R2, A2B_R2_var = G_A2R2(A_abs, training=False) # Mean R2s
        else:
            A2B_R2 = G_A2R2(A_abs, training=False)
            A2B_R2_var = None
        A2B_R2 = tf.where(A[:,:,:,:1]!=0.0,A2B_R2,0.0)

        # Compute FM from complex-valued images
        if args.UQ:
            _, A2B_FM, A2B_var = G_A2B(A, training=False)
            A2B_FM_var = tf.where(A[:,:,:,:1]!=0.0,A2B_var,0.0)
        else:
            A2B_FM = G_A2B(A, training=False)
            A2B_FM_var = None
        A2B_FM = tf.where(A[:,:,:,:1]!=0.0,A2B_FM,0.0)
        A2B_PM = tf.concat([A2B_R2,A2B_FM], axis=-1)

        # Magnitude of water/fat images
        A2B_WF = wf.get_rho(A,A2B_PM,complex_data=(args.G_model=='complex'))
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))

        A2B = tf.concat([A2B_WF,A2B_R2,A2B_FM], axis=-1)
        A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM], axis=-1)

        # Variance map mask
        if args.UQ:
            A2B_PM_var = tf.concat([A2B_R2_var,A2B_FM_var], axis=-1)
            A2B_PM_var = tf.where(A2B_PM_var!=0.0,A2B_PM_var,1e-12)
            A2B_PM_var = tf.where(A[:,:,:,:2]!=0.0,A2B_PM_var,1e-12)
            A2B_WF_var = wf.PDFF_uncertainty(A,A2B,A2B_PM_var)
            A2B_WF_var = tf.where(A2B_WF_var!=0.0,A2B_WF_var,1e-12)
            A2B_WF_var = tf.where(A[:,:,:,:4]!=0.0,A2B_WF_var,1e-12)
            A2B_var = tf.concat([A2B_WF_var,A2B_PM_var], axis=-1)
        else:
            A2B_var = None

    return A2B_abs, A2B_PM_var


# run
save_dir = py.join(args.experiment_dir, 'samples_testing', 'A2B')
py.mkdir(save_dir)
i = 0

for A, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    B = tf.expand_dims(B,axis=0)
    A2B, A2B_PM_var = sample(A,B)

    r2_aux = np.squeeze(A2B[:,2,:,:,1])
    field_aux = np.squeeze(A2B[:,2,:,:,0])

    if args.UQ:
        # Get water/fat uncertainties
        w_aux = np.squeeze(tf.abs(tf.complex(A2B[:,0,:,:,0],A2B[:,0,:,:,1])))
        f_aux = np.squeeze(tf.abs(tf.complex(A2B[:,1,:,:,0],A2B[:,1,:,:,1])))
        W_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,0,:,:,0],A2B_var[:,0,:,:,1])))
        F_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,1,:,:,0],A2B_var[:,1,:,:,1])))
        r2s_var = np.squeeze(A2B_var[:,2,:,:,1])*(r2_sc**2)
        field_var = np.squeeze(A2B_var[:,2,:,:,0])*(fm_sc**2)
        hgt_plt, wdt_plt, nr, nc = 10, 20, 3, 5
    else:
        w_aux = np.squeeze(A2B[:,:,:,0])
        f_aux = np.squeeze(A2B[:,:,:,1])
        hgt_plt, wdt_plt, nr, nc = 9, 16, 2, 3

    # (Mean) PDFF estimation
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0

    wn_aux = np.squeeze(np.abs(tf.complex(B[:,0,:,:,0],B[:,0,:,:,1])))
    fn_aux = np.squeeze(np.abs(tf.complex(B[:,1,:,:,0],B[:,1,:,:,1])))
    r2n_aux = np.squeeze(B[:,2,:,:,1])
    fieldn_aux = np.squeeze(B[:,2,:,:,0])
    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0
    
    if i%args.n_plot == 0 or args.dataset == 'phantom':
        fig,axs=plt.subplots(figsize=(wdt_plt, hgt_plt), nrows=nr, ncols=nc)

        # Ground-truth maps in the first row
        FF_ok = axs[0,0].imshow(PDFFn_aux, cmap='jet',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(FF_ok, ax=axs[0,0])
        axs[0,0].axis('off')

        # Estimated maps in the second row
        FF_est =axs[1,0].imshow(PDFF_aux, cmap='jet',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(FF_est, ax=axs[1,0])
        axs[1,0].axis('off')

        if args.UQ:
            # Ground-truth maps
            W_ok =  axs[0,1].imshow(wn_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[0,1])
            axs[0,1].axis('off')

            F_ok =  axs[0,2].imshow(fn_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[0,2])
            axs[0,2].axis('off')

            r2_ok = axs[0,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[0,3])
            axs[0,3].axis('off')

            field_ok =  axs[0,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[0,4])
            axs[0,4].axis('off')

            # Estimated images/maps
            W_est = axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_est, ax=axs[1,1])
            axs[1,1].axis('off')

            F_est = axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_est, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_est= axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_est, ax=axs[1,3])
            axs[1,3].axis('off')

            field_est = axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_est, ax=axs[1,4])
            axs[1,4].axis('off')

            # Uncertainty maps in the 3rd row
            fig.delaxes(axs[2,0]) # No PDFF variance map

            W_uq = axs[2,1].matshow(W_var, cmap='gnuplot2',
                                    norm=LogNorm(vmin=1e-12,vmax=1e-8))
            fig.colorbar(W_uq, ax=axs[2,1])
            axs[2,1].axis('off')

            F_uq = axs[2,2].matshow(F_var, cmap='gnuplot2',
                                    norm=LogNorm(vmin=1e-12,vmax=1e-8))
            fig.colorbar(F_uq, ax=axs[2,2])
            axs[2,2].axis('off')

            r2s_uq=axs[2,3].matshow(r2s_var, cmap='gnuplot',
                                    norm=LogNorm(vmin=10,vmax=0.1))
            fig.colorbar(r2s_uq, ax=axs[2,3])
            axs[2,3].axis('off')

            field_uq = axs[2,4].matshow(field_var, cmap='gnuplot2',
                                        norm=LogNorm(vmin=10,vmax=0.1))
            fig.colorbar(field_uq, ax=axs[2,4])
            axs[2,4].axis('off')
        else:
            # Ground truth
            r2_ok = axs[0,1].imshow(r2n_aux, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[0,1])
            axs[0,1].axis('off')

            field_ok =  axs[0,2].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[0,2])
            axs[0,2].axis('off')

            # Ground truth
            r2_est =axs[1,1].imshow(r2_aux, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_est, ax=axs[1,1])
            axs[1,1].axis('off')

            field_est = axs[1,2].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_est, ax=axs[1,2])
            axs[1,2].axis('off')
        
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