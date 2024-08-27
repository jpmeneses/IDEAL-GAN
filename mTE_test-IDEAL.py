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
from matplotlib.colors import LogNorm

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--dataset', type=str, default='multiTE', choices=['multiTE','3ech','JGalgani','phantom_1p5','phantom_3p0'])
py.arg('--data_size', type=int, default=384, choices=[192,384])
py.arg('--te_input', type=bool, default=True)
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

if not(hasattr(args,'UQ')):
    py.arg('--UQ', type=bool, default=False)
    py.arg('--UQ_R2s', type=bool, default=False)
    py.arg('--UQ_calib', type=bool, default=False)
    UQ_args = py.args()
    args.__dict__.update(UQ_args.__dict__)

if not(hasattr(args,'G_model')):
    py.arg('--G_model', default='U-Net', choices=['multi-decod','U-Net','MEBCRN'])
    GM_args = py.args()
    args.__dict__.update(GM_args.__dict__)


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

r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
if args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
    dataset_hdf5 = args.dataset + '_RF_192_128_complex_2D.hdf5'
else:
    dataset_hdf5 = args.dataset + '_GC_' + str(args.data_size) + '_complex_2D.hdf5'

if args.dataset == 'JGalgani':
    num_slice_list = [0,21,20,24,24,21,24,21,21,24,21,21,22,27,23,22,20,24,21,21,22,20]
    rnc = True
elif args.dataset == 'multiTE':
    ini_idxs = [0,84,204,300,396,484,580,680,776,848,932,1028, 1100,1142,1190,1232,1286,1334,1388,1460]
    delta_idxs = [21,24,24,24,22,24,25,24,18,21,24,18, 21,24,21,18,16,18,24,21]
    # First Patient
    if args.TE1 == 0.0014 and args.dTE == 0.0022:
        k_idxs = [(0,1),(2,3)]
    elif args.TE1 == 0.0013 and args.dTE == 0.0023:
        k_idxs = [(0,1),(3,4)]
    else:
        k_idxs = [(0,2)]
    for k in k_idxs:
        custom_list = [a for a in range(ini_idxs[0]+k[0]*delta_idxs[0],ini_idxs[0]+k[1]*delta_idxs[0])]
    # Rest of the patients
    for i in range(1,len(ini_idxs)):
        if (i<=11) and args.TE1 == 0.0013 and args.dTE == 0.0022:
            k_idxs = [(0,1),(2,3)]
        elif (i<=11) and args.TE1 == 0.0014 and args.dTE == 0.0022:
            k_idxs = [(0,1),(3,4)]
        elif (i==1) and args.TE1 == 0.0013 and args.dTE == 0.0023:
            k_idxs = [(0,1),(4,5)]
        elif (i==15 or i==16) and args.TE1 == 0.0013 and args.dTE == 0.0023:
            k_idxs = [(0,1),(2,3)]
        elif (i>=17) and args.TE1 == 0.0013 and args.dTE == 0.0024:
            k_idxs = [(0,1),(2,3)]
        else:
            k_idxs = [(0,2)]
        for k in k_idxs:
            custom_list += [a for a in range(ini_idxs[i]+k[0]*delta_idxs[i],ini_idxs[i]+k[1]*delta_idxs[i])]
else:
    num_slice_list = None
    rnc = False

if args.dataset == 'JGalgani' or args.dataset == '3ech':
    testX, testY=data.load_hdf5(dataset_dir,dataset_hdf5,num_slice_list=num_slice_list,remove_non_central=rnc,
                                acqs_data=True,te_data=False,remove_zeros=True,MEBCRN=True)
    TEs = np.ones((testX.shape[0],1),dtype=np.float32)
elif args.dataset == 'multiTE':
    testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, custom_list=custom_list,
                                        acqs_data=True,te_data=True,remove_zeros=False,MEBCRN=True)
else:
    testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, acqs_data=True, 
                                        te_data=True,remove_zeros=True,MEBCRN=True)
if args.dataset == 'multiTE':
    testX, testY, TEs = data.group_TEs(testX,testY,TEs,TE1=args.TE1,dTE=args.dTE,MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(testX)
_,n_out,_,_,_ = np.shape(testY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',ne)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)
print('TEs shape:',TEs.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,TEs,testY))
A_B_dataset_test.batch(1)

# model
if args.ME_layer:
    input_shape = (None,None,None,n_ch)
else:
    input_shape = (None,None,2*ne)

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
                                te_shape=(None,),
                                ME_layer=args.ME_layer,
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WF-PM':
        n_out = 4
    elif args.out_vars == 'FM' or args.out_vars == 'R2s':
        n_out = 1
    else:
        n_out = 2
    G_A2B = dl.UNet(input_shape=(None,hgt,wdt,n_ch),
                    n_out=n_out,
                    bayesian=args.UQ,
                    ME_layer=args.ME_layer,
                    te_input=args.te_input,
                    te_shape=(None,),
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
    if args.out_vars == 'R2s':
        G_A2R2= dl.UNet(input_shape=(None,hgt,wdt,1),
                        bayesian=args.UQ_R2s,
                        ME_layer=args.ME_layer,
                        te_input=args.te_input,
                        te_shape=(None,),
                        filters=args.n_G_filters,
                        output_activation='sigmoid',
                        self_attention=args.D2_SelfAttention)
        G_calib = tf.keras.Sequential()
        G_calib.add(tf.keras.layers.Conv2D(1,1,use_bias=False,kernel_initializer='ones',kernel_constraint=tf.keras.constraints.NonNeg()))
        G_calib.build((None, 1, hgt, wdt, 1))

elif args.G_model == 'MEBCRN':
    if args.out_vars == 'WFc':
        n_out = 4
        out_activ = None
    else:
        n_out = 2
        out_activ = 'sigmoid'
    G_A2B=dl.MEBCRN(input_shape=(ne,hgt,wdt,2),
                    n_outputs=n_out,
                    output_activation=out_activ,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

# restore
if args.out_vars == 'R2s':
    tl.Checkpoint(dict(G_A2B=G_A2B,G_A2R2=G_A2R2,G_calib=G_calib), py.join(args.experiment_dir, 'checkpoints')).restore()
else:
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
            if TE is None:
                TE = wf.gen_TEvar(ne, bs=A.shape[0], orig=True) # (nb,ne,1)
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
            if TE is None:
                TE = wf.gen_TEvar(ne, bs=A.shape[0], orig=True) # (nb,ne,1)
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
            if TE is None:
                TE = wf.gen_TEvar(ne, bs=A.shape[0], orig=True) # (nb,ne,1)
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
    elif args.out_vars == 'R2s':
        A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
        # Compute R2s maps using only-mag images
        A2B_R2 = G_A2R2(A_abs, training=False) # Mean R2s
        if args.UQ_R2s:
            A2B_R2_nu = A2B_R2.mean()
            if args.UQ_calib:
                A2B_R2_sigma = G_calib(A2B_R2.stddev(), training=False)
            else:
                A2B_R2_sigma = A2B_R2.stddev()
        else:
            A2B_R2_nu = tf.zeros_like(A2B_R2)
            A2B_R2_sigma = tf.zeros_like(A2B_R2)

        # Compute FM from complex-valued images
        A2B_FM = G_A2B(A, training=False)
        if args.UQ:
            A2B_FM_var = A2B_FM.stddev()
        else:
            A2B_FM_var = tf.zeros_like(A2B_FM)
        A2B_PM = tf.concat([A2B_FM.mean(),A2B_R2.mean()], axis=-1)

        # Variance map mask
        if args.UQ:
            A2B_WF, A2B_WF_var = wf.PDFF_uncertainty(A, A2B_FM, A2B_R2, te=TE, rem_R2=False)
            A2B_WF_var = tf.concat([A2B_WF_var,tf.zeros_like(A2B_WF_var)],axis=-1)
            A2B_PM_var = tf.concat([A2B_FM.variance(),A2B_R2.variance()],axis=-1)
            A2B_var = tf.concat([A2B_WF_var,A2B_PM_var], axis=1)
            A2B_var = tf.where(A[:,:5,:,:,:]!=0,A2B_var,1e-8)
        else:
            A2B_WF = wf.get_rho(A,A2B_PM)
            A2B_var = None

        A2B = tf.concat([A2B_WF,A2B_PM], axis=1)
        A2B = tf.where(A[:,:3,:,:,:]!=0,A2B,0.0)

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
    plot_hgt = 9
else:
    plot_hgt = 13

for A, TE_smp, B in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
    A = tf.expand_dims(A,axis=0)
    TE_smp = tf.expand_dims(TE_smp,axis=0)
    B = tf.expand_dims(B,axis=0)
    B_WF = B[:,:2,:,:,:]
    B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))
    
    if args.dataset == 'JGalgani' or args.dataset == '3ech':
        A2B, A2B_var = sample(A, B)
    else:
        A2B, A2B_var = sample(A, B, TE_smp)
    A2B_WF = A2B[:,:2,:,:,:]
    A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))

    w_aux = np.squeeze(tf.abs(tf.complex(A2B[:,0,:,:,0],A2B[:,0,:,:,1])))
    f_aux = np.squeeze(tf.abs(tf.complex(A2B[:,1,:,:,0],A2B[:,1,:,:,1])))
    r2_aux = np.squeeze(A2B[:,2,:,:,1])*r2_sc
    field_aux = np.squeeze(A2B[:,2,:,:,0])*fm_sc
    PDFF_aux = f_aux/(w_aux+f_aux)
    PDFF_aux[np.isnan(PDFF_aux)] = 0.0
    if args.UQ:
        # Get water/fat uncertainties
        W_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,0,:,:,0],A2B_var[:,0,:,:,1])))
        WF_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,1,:,:,0],A2B_var[:,1,:,:,1])))
        F_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,3,:,:,0],A2B_var[:,3,:,:,1])))
        r2s_var = np.squeeze(A2B_var[:,-1,:,:,1])*(r2_sc**2)
        field_var = np.squeeze(A2B_var[:,-1,:,:,0])*(fm_sc**2)
        field_var = np.where(field_var!=((fm_sc**2)*1e-8),field_var,1e-5)

    wn_aux = np.squeeze(B_WF_abs[:,0,:,:,:])
    fn_aux = np.squeeze(B_WF_abs[:,1,:,:,:])
    r2n_aux = np.squeeze(B[:,2,:,:,1])*r2_sc
    fieldn_aux = np.squeeze(B[:,2,:,:,0])*fm_sc
    PDFFn_aux = fn_aux/(wn_aux+fn_aux)
    PDFFn_aux[np.isnan(PDFFn_aux)] = 0.0
    
    if i%args.n_plot == 0 or args.dataset == 'phantom_1p5' or args.dataset == 'phantom_3p0':
        if args.UQ:
            fig,axs=plt.subplots(figsize=(20, 10), nrows=3, ncols=5)

            # Ground-truth maps in the first row
            FF_ok = axs[0,0].imshow(PDFF_aux, cmap='jet',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(FF_ok, ax=axs[0,0])
            axs[0,0].axis('off')

            # Estimated maps in the second row
            FF_est =axs[1,0].imshow(PDFF_aux-PDFFn_aux, cmap='plasma',
                                    interpolation='none', vmin=-0.15, vmax=0.15)
            fig.colorbar(FF_est, ax=axs[1,0])
            axs[1,0].axis('off')

            # Estimated maps
            W_ok =  axs[0,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[0,1])
            axs[0,1].axis('off')

            F_ok =  axs[0,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[0,2])
            axs[0,2].axis('off')

            r2_ok = axs[0,3].imshow(r2_aux, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[0,3])
            axs[0,3].axis('off')

            field_ok =  axs[0,4].imshow(field_aux, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[0,4])
            axs[0,4].axis('off')

            # Error w.r.t. reference images/maps
            W_est = axs[1,1].imshow(w_aux-wn_aux, cmap='plasma',
                                    interpolation='none', vmin=-0.15, vmax=0.15)
            fig.colorbar(W_est, ax=axs[1,1])
            axs[1,1].axis('off')

            F_est = axs[1,2].imshow(f_aux-fn_aux, cmap='plasma',
                                    interpolation='none', vmin=-0.15, vmax=0.15)
            fig.colorbar(F_est, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_est= axs[1,3].imshow(r2_aux-r2n_aux, cmap='plasma',
                                    interpolation='none', vmin=-r2_sc/5, vmax=r2_sc/5)
            fig.colorbar(r2_est, ax=axs[1,3])
            axs[1,3].axis('off')

            field_est = axs[1,4].imshow(field_aux-fieldn_aux, cmap='plasma',
                                        interpolation='none', vmin=-r2_sc/5, vmax=r2_sc/5)
            fig.colorbar(field_est, ax=axs[1,4])
            axs[1,4].axis('off')

            # Uncertainty maps in the 3rd row
            # Plot PDFF absolute error instead of unavailable uncertainty
            WF_uq = axs[2,0].matshow(WF_var, cmap='gnuplot2',
                                    norm=LogNorm(vmin=1e-2,vmax=1e0))
            fig.colorbar(WF_uq, ax=axs[2,0])
            axs[2,0].axis('off')

            W_uq = axs[2,1].matshow(W_var, cmap='gnuplot2',
                                    norm=LogNorm(vmin=1e-2,vmax=1e0))
            fig.colorbar(W_uq, ax=axs[2,1])
            axs[2,1].axis('off')

            F_uq = axs[2,2].matshow(F_var, cmap='gnuplot2',
                                    norm=LogNorm(vmin=1e-2,vmax=1e0))
            fig.colorbar(F_uq, ax=axs[2,2])
            axs[2,2].axis('off')

            if args.out_vars == 'WF' or args.out_vars == 'FM':
                fig.delaxes(axs[2,3]) # No R2s variance map
            else:
                r2s_uq=axs[2,3].matshow(r2s_var, cmap='gnuplot',
                                        norm=LogNorm(vmin=1e0,vmax=1e3))
                fig.colorbar(r2s_uq, ax=axs[2,3])
                axs[2,3].axis('off')

            field_uq = axs[2,4].matshow(field_var, cmap='gnuplot2',
                                        norm=LogNorm(vmin=1e-5,vmax=1e-2))
            fig.colorbar(field_uq, ax=axs[2,4])
            axs[2,4].axis('off')
        else:
            fig,axs=plt.subplots(figsize=(16, plot_hgt), nrows=3, ncols=3)

            # A2B maps in the first row
            F_ok = axs[0,0].imshow(PDFF_aux, cmap='jet',
                              interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[0,0]).ax.tick_params(labelsize=20)
            axs[0,0].axis('off')

            r2_ok = axs[0,1].imshow(r2_aux, cmap=cmap,
                                    interpolation='none', vmin=0, vmax=lmax)
            fig.colorbar(r2_ok, ax=axs[0,1]).ax.tick_params(labelsize=20)
            axs[0,1].axis('off')

            field_ok = axs[0,2].imshow(field_aux, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[0,2]).ax.tick_params(labelsize=20)
            axs[0,2].axis('off')

            # Ground-truth in the third row
            F_unet = axs[1,0].imshow(PDFFn_aux, cmap='jet',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[1,0]).ax.tick_params(labelsize=20)
            axs[1,0].axis('off')

            r2_unet=axs[1,1].imshow(r2n_aux, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[1,1]).ax.tick_params(labelsize=20)
            axs[1,1].axis('off')

            field_unet =axs[1,2].imshow(fieldn_aux, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_unet, ax=axs[1,2]).ax.tick_params(labelsize=20)
            axs[1,2].axis('off')

            F_err = axs[2,0].imshow(np.abs(PDFF_aux-PDFFn_aux), cmap='gray',
                              interpolation='none', vmin=0, vmax=.1)
            fig.colorbar(F_err, ax=axs[2,0]).ax.tick_params(labelsize=20)
            axs[2,0].axis('off')

            r2_err = axs[2,1].imshow(np.abs(r2_aux-r2n_aux), cmap='pink',
                                    interpolation='none', vmin=0, vmax=lmax/5)
            fig.colorbar(r2_err, ax=axs[2,1]).ax.tick_params(labelsize=20)
            axs[2,1].axis('off')

            field_err = axs[2,2].imshow(np.abs(field_aux-fieldn_aux), cmap='bone',
                                        interpolation='none', vmin=0, vmax=fm_sc/20)
            fig.colorbar(field_err, ax=axs[2,2]).ax.tick_params(labelsize=20)
            axs[2,2].axis('off')

        if args.dataset == 'JGalgani' or args.dataset == '3ech':
            fig.suptitle('TE1/dTE: '+str([args.TE1,args.dTE]), fontsize=18)
        else:
            fig.suptitle('TE1/dTE: '+str([TE_smp[0,0,0].numpy(),np.mean(np.diff(TE_smp,axis=1))]), fontsize=18)

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
    
    if args.dataset == 'JGalgani' or args.dataset == '3ech':
        ws_MSE.write(i+1,5,args.TE1)
        ws_MSE.write(i+1,6,args.dTE)
    else:
        ws_MSE.write(i+1,5,TE_smp[0,0,0].numpy())
        ws_MSE.write(i+1,6,np.mean(np.diff(TE_smp,axis=1)))

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
    # w_ssim = tf.image.ssim(w_aux,wn_aux,1)
    # f_ssim = tf.image.ssim(f_aux,fn_aux,1)
    # pdff_ssim = tf.image.ssim(PDFF_aux,PDFFn_aux,1)
    # r2_ssim = tf.image.ssim(r2_aux/r2_sc,r2n_aux/r2_sc,1)
    # fm_ssim = tf.image.ssim(field_aux/fm_sc,fieldn_aux/fm_sc,1)

    # ws_SSIM.write(i+1,0,w_ssim)
    # ws_SSIM.write(i+1,1,f_ssim)
    # ws_SSIM.write(i+1,2,pdff_ssim)
    # ws_SSIM.write(i+1,3,r2_ssim)
    # ws_SSIM.write(i+1,4,fm_ssim)
    
    i += 1

workbook.close()