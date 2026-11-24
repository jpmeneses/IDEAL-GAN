import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib

from itertools import cycle
from time import process_time

import tf2lib as tl
import DLlib as dl
import pylib as py
import wflib as wf
import data

py.arg('--experiment_dir', default='Mag-500')
py.arg('--test_data', default='DICOM', choices=['DICOM','NIFTI'])
py.arg('--model_sel', default='Mag', choices=['U-Net','MDWF-Net','2D-Net','VET-Net','AI-DEAL','Single','Mag'])
py.arg('--echo_sel', type=int, default=12)
py.arg('--phase_const', type=bool, default=True)
py.arg('--pdff_mag', type=bool, default=True)
test_args = py.args()

experiment_dir = py.join('output',test_args.experiment_dir)
args = py.args_from_yaml(py.join(experiment_dir, 'settings.yml'))

output_dir = py.join(experiment_dir,'out-val')
py.mkdir(output_dir)

fm_sc = 300.0
r2_sc = 200.0

if test_args.test_data == 'DICOM':
    folders = [os.path.join(args.dataset_dir, d) for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]
    folders_mr = [os.path.join(f, os.listdir(f)[0]) for i, f in enumerate(folders) if os.path.join(f, os.listdir(f)[0])]
    folders_cse = list()
    for f in folders_mr:
        scan_files = os.listdir(f)
        if test_args.test_data == 'DICOM':
            cse_scan = [item for item in scan_files if "MECSE" in item]
        elif test_args.test_data == 'NIFTI':
            cse_scan = [item for item in scan_files if "nifti" in item]
        folders_cse.append(os.path.join(f,cse_scan[0]))
    idxs = [i for i in range(len(folders_cse))]
    n_folds = 5
    n_divs = int(np.ceil(len(folders_cse)/n_folds))
    idxs_val = [i for i in range(n_divs*(args.k_fold-1), min(n_divs*args.k_fold, len(folders_cse)))]
    del idxs[n_divs*(args.k_fold-1):n_divs*args.k_fold]
    print(idxs)
    print(idxs_val)

    A_dataset_val = tf.data.Dataset.from_tensor_slices([folders_cse[idx] for idx in idxs_val])
    if test_args.test_data == 'DICOM':
        A_dataset_val = A_dataset_val.map(lambda f: data.tf_load_dicom_series(f))
    elif test_args.test_data == 'NIFTI':
        A_dataset_val = A_dataset_val.map(lambda f: data.tf_load_nifti_series(f))
    # A_dataset_val = A_dataset_val.unbatch()
    # len_val = sum(1 for _ in A_B_dataset_val)
    # A_B_dataset_val = A_B_dataset_val.batch(1)

if test_args.model_sel == 'U-Net':
    G_A2B = dl.UNet(input_shape=(None,None,2*test_args.echo_sel), n_out=2, filters=args.n_G_filters,output_activation='relu')
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == 'MDWF-Net':
    G_A2B = dl.MDWF_Generator(input_shape=(None,None,2*test_args.echo_sel), filters=args.n_G_filters, 
                              WF_self_attention=args.D1_SelfAttention, R2_self_attention=args.D2_SelfAttention,
                              FM_self_attention=args.D3_SelfAttention)
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == '2D-Net':
    G_A2B = dl.PM_Generator(input_shape=(None,None,2*test_args.echo_sel), te_input=False, ME_layer=None, filters=args.n_G_filters)
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == 'VET-Net':
    G_A2B = dl.PM_Generator(input_shape=(None,None,None,2), te_input=True, te_shape=(None,), filters=args.n_G_filters)
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == 'AI-DEAL':
    G_A2B = dl.UNet(input_shape=(None,None,None,2), bayesian=True, ME_layer=True, filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)
    G_A2R2= dl.UNet(input_shape=(None,None,None,1), bayesian=True, ME_layer=True, filters=args.n_G_filters,
                    output_activation='sigmoid', self_attention=args.D2_SelfAttention)
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B, G_A2R2=G_A2R2), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == 'Single':
    G_mag = dl.UNet(input_shape=(None,None,None,1),n_out=3,ME_layer=True,filters=args.n_G_filters,
                output_activation='sigmoid',self_attention=args.D1_SelfAttention)
    G_pha = dl.UNet(input_shape=(None,None,None,1),n_out=4,ME_layer=True,filters=args.n_G_filters,
                    output_activation='linear',self_attention=args.D2_SelfAttention)
    checkpoint = tl.Checkpoint(dict(G_mag=G_mag, G_pha=G_pha), py.join(experiment_dir, 'checkpoints'))
elif test_args.model_sel == 'Mag':
    G_mag = dl.UNet(input_shape=(None,None,None,1),bayesian=(args.main_loss=='Rice'),ME_layer=True,filters=args.n_G_filters,
                    te_input=(args.training_mode=='supervised'),te_shape=(None,),
                    output_activation='sigmoid',self_attention=args.D1_SelfAttention)
    checkpoint = tl.Checkpoint(dict(G_mag=G_mag), py.join(experiment_dir, 'checkpoints'))

try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# Sample function
@tf.function
def sample(A, TE):
    if A.shape[1] > test_args.echo_sel:
        A = A[:,:test_args.echo_sel,...]
        TE = TE[:,:test_args.echo_sel,...]
    A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A), axis=-1, keepdims=True))
    A_msk = tf.reduce_mean(A_abs, axis=1, keepdims=True)
    A2B_msk = tf.concat([A_msk,A_msk], axis=-1)
    if test_args.model_sel == 'U-Net':
        A_pf = data.A_from_MEBCRN(A) # CHANGE TO NON-MEBCRN FORMAT
        A2B_WF_abs = G_A2B(A_pf, training=False)
        A2B_WF_abs = tf.expand_dims(A2B_WF_abs, axis=1)
        A2B_WF_abs = tf.transpose(A2B_WF_abs, perm=[0,4,2,3,1])
        A2B_WF = tf.concat([A2B_WF_abs, tf.zeros_like(A2B_WF_abs)], axis=-1)
        A2B = tf.concat([A2B_WF, tf.zeros_like(A2B_WF[:,:1,...])], axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
        A2B_var = None
    elif test_args.model_sel == 'MDWF-Net':
        A_pf = data.A_from_MEBCRN(A) # CHANGE TO NON-MEBCRN FORMAT
        A2B = G_A2B(A_pf, training=False)
        A2B = tf.expand_dims(A2B, axis=1)
        A2B_PM = A2B[...,-1:-3:-1]
        A2B_WF_abs = A2B[...,:2]
        A2B_WF_abs = tf.transpose(A2B_WF_abs, perm=[0,4,2,3,1])
        A2B_WF = tf.concat([A2B_WF_abs, tf.zeros_like(A2B_WF_abs)], axis=-1)
        A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
        A2B_var = None
    elif test_args.model_sel == '2D-Net':
        A_pf = data.A_from_MEBCRN(A) # CHANGE TO NON-MEBCRN FORMAT
        A2B_PM = G_A2B(A_pf, training=False)
        A2B_PM = tf.expand_dims(A2B_PM, axis=1)
        A2B_PM = A2B_PM[...,::-1]
        A2B_WF = wf.get_rho(A, A2B_PM, te=TE, phase_constraint=test_args.phase_const)
        A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
        A2B_var = None
    elif test_args.model_sel == 'VET-Net':
        A2B_PM = G_A2B([A,TE], training=False) #[:,:test_args.echo_sel,...]
        A2B_WF = wf.get_rho(A, A2B_PM, field=args.field, te=TE, phase_constraint=test_args.phase_const)
        A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
        A2B_var = None
    elif test_args.model_sel == 'AI-DEAL':
        A2B_var_msk = tf.concat([A2B_msk,A2B_msk,A2B_msk,A2B_msk,A2B_msk], axis=1)
        A2B_msk = tf.concat([A2B_msk,A2B_msk,A2B_msk], axis=1)
        A2B_FM = G_A2B(A, training=False)
        A2B_R2 = G_A2R2(A_abs, training=False)
        A2B_PM = tf.concat([A2B_FM.mean(),A2B_R2.mean()],axis=-1)
        A2B_WF, A2B_WF_var = wf.PDFF_uncertainty(A, A2B_FM, A2B_R2, te=TE, rem_R2=False)
        A2B_WF_var = tf.concat([A2B_WF_var,tf.zeros_like(A2B_WF_var)],axis=-1)
        A2B_PM_var = tf.concat([A2B_FM.variance(),A2B_R2.sigma],axis=-1)
        A2B_var = tf.concat([A2B_WF_var,A2B_PM_var], axis=1)
        A2B_var = tf.where(A2B_var_msk>=5e-2, A2B_var, 1e-12)
        A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
    elif test_args.model_sel == 'Single':
        A_pha = tf.math.atan2(A[...,1:],A[...,:1]) / np.pi
        A2B_abs = G_mag(A_abs, training=False)
        A2B_pha = G_pha(A_pha, training=False)
        A2B_WF_abs = A2B_abs[...,:2]
        A2B_WF_pha = A2B_pha[...,:2]
        A2B_WF_r = A2B_WF_abs * tf.math.cos(4*np.pi*A2B_WF_pha)
        A2B_WF_i = A2B_WF_abs * tf.math.sin(4*np.pi*A2B_WF_pha)
        A2B_WF = tf.concat([A2B_WF_r,A2B_WF_i],axis=1)
        A2B_WF = tf.transpose(A2B_WF,perm=[0,4,2,3,1])
        A2B_PM = tf.concat([A2B_pha[...,2:3],A2B_abs[...,2:]],axis=-1)
        A2B_bip = tf.concat([A2B_pha[...,-1:],tf.zeros_like(A2B_pha[...,-1:])],axis=-1)
        A2B = tf.concat([A2B_WF,A2B_PM,A2B_bip],axis=1)
        A2B_bip_msk = tf.repeat(A_msk,4,axis=1)
        A2B = tf.where(A2B_bip_msk>=5e-2, A2B, 0.0)
        A2B_var = None
    elif test_args.model_sel == 'Mag':
        if args.main_loss == 'Rice':
            if args.training_mode == 'supervised' and args.n_echoes == 0:
                A2B_R2_prob = G_mag([A_abs, TE], training=False)
            else:
                A2B_R2_prob = G_mag(A_abs, training=False)
            A2B_R2 = A2B_R2_prob.mean()
        elif args.training_mode == 'supervised':
            A2B_R2 = G_mag([A_abs, TE], training=False)
        else:
            A2B_R2 = G_mag(A_abs, training=False)
        A2B_R2 = tf.where(A_abs[:,:1,...]!=0.0,A2B_R2,0.0)
        A2B_WF_abs, A2B2A_abs, A2B_WF_var, _ = wf.CSE_mag(A_abs, A2B_R2, [args.field, TE], r2_sc=r2_sc, uncertainty=True) # r2_sc=r2_sc
        A2B2A_abs = tf.where(A_abs!=0.0,A2B2A_abs,0.0)
        A2B_PM = tf.concat([tf.zeros_like(A2B_R2),A2B_R2],axis=-1)
        A2B_WF = tf.concat([A2B_WF_abs,tf.zeros_like(A2B_WF_abs)],axis=-1)
        A2B = tf.concat([A2B_WF,A2B_PM],axis=1)
        A2B = tf.where(A2B_msk>=5e-2, A2B, 0.0)
        A2B_WF_var = tf.concat([A2B_WF_var,tf.zeros_like(A2B_WF_var)], axis=-1)
        if args.main_loss == 'Rice':
            A2B_PM_var = tf.concat([tf.zeros_like(A2B_R2_prob.variance()),A2B_R2_prob.variance()], axis=-1)
            A2B_var = tf.concat([tf.zeros_like(A2B_WF_var),A2B_WF_var,A2B_WF_var,
                                 tf.zeros_like(A2B_WF_var),A2B_PM_var], axis=1)
        else:
            A2B_var = tf.concat([tf.zeros_like(A2B_WF_var),A2B_WF_var,A2B_WF_var,
                                 tf.zeros_like(A2B_WF_var),tf.zeros_like(A2B_WF_var)], axis=1)
    return A2B, A2B_var


def test(A, TE=None):
    A2B, A2B_var = sample(A, TE)
    return A2B, A2B_var


i = 0

for A_vol in A_dataset_val:
    listdir = output_dir.split('/')
    listdir.append('res_MP_' + ''.join(test_args.experiment_dir.split('-')) + '_vol-%03d' % i)
    
    # Future output variables
    F = list() # PDFF 
    P = list() # FM 
    R = list() # WF 
    R2= list() # R2 

    # Future uncertainty variables (AI-DEAL only)
    PDFF_vars = list()
    WF_vars = list()
    R2_vars = list()
    FM_vars = list()

    te_var=wf.gen_TEvar(test_args.echo_sel, bs=1, TE_ini_min=0.879e-3,
                        TE_ini_d=None, d_TE_min=0.662e-3, d_TE_d=None)
    
    for A in A_vol:
        A = tf.expand_dims(A,axis=0)
        A2B, A2B_var = test(A, te_var)
        
        # CSE-MR images at each echo
        mag_ech1 = np.squeeze(np.abs(tf.complex(A[:,0,:,:,0],A[:,0,:,:,1])))
        pha_ech1 = np.squeeze(np.arctan2(A[:,0,:,:,1],A[:,0,:,:,0]))
        mag_ech2 = np.squeeze(np.abs(tf.complex(A[:,1,:,:,0],A[:,1,:,:,1])))
        pha_ech2 = np.squeeze(np.arctan2(A[:,1,:,:,1],A[:,1,:,:,0]))
        mag_ech3 = np.squeeze(np.abs(tf.complex(A[:,2,:,:,0],A[:,2,:,:,1])))
        pha_ech3 = np.squeeze(np.arctan2(A[:,2,:,:,1],A[:,2,:,:,0]))
        
        # Estimated quantitative maps
        w_m_aux = np.squeeze(tf.abs(tf.complex(A2B[:,0,:,:,:1],A2B[:,0,:,:,1:])),axis=0)
        w_p_aux = np.squeeze(tf.math.atan2(A2B[:,0,:,:,1],A2B[:,0,:,:,0]))
        f_m_aux = np.squeeze(tf.abs(tf.complex(A2B[:,1,:,:,:1],A2B[:,1,:,:,1:])),axis=0)
        f_p_aux = np.squeeze(tf.math.atan2(A2B[:,1,:,:,1],A2B[:,1,:,:,0]))
        wf_m_aux = np.squeeze(tf.abs(tf.complex(A2B[:,0,:,:,:1]+A2B[:,1,:,:,:1],A2B[:,0,:,:,1:]+A2B[:,1,:,:,1:])),axis=0)
        r2_aux = np.squeeze(A2B[:,2,:,:,1])
        field_aux = np.squeeze(A2B[:,2,:,:,0])
        if test_args.pdff_mag:
            PDFF_aux = f_m_aux/(w_m_aux+f_m_aux)
        else:
            PDFF_aux = np.where(f_m_aux >= w_m_aux, f_m_aux/wf_m_aux, 1-(w_m_aux/wf_m_aux))
        PDFF_aux[np.isnan(PDFF_aux)] = 0.0
        # Update maps to be saved in MAT file
        F.append(PDFF_aux*100.0)
        P.append(np.expand_dims(field_aux*fm_sc, axis=-1))
        R.append(np.expand_dims(np.concatenate([w_m_aux,f_m_aux],axis=-1), axis=-2))
        R2.append(np.expand_dims(r2_aux*r2_sc, axis=-1))
        
        # Estimated uncertainty maps (if available)
        if A2B_var is not None:
            W_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,0,:,:,:1],A2B_var[:,0,:,:,1:])),axis=0)
            WF_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,1,:,:,:1],A2B_var[:,1,:,:,1:])),axis=0)
            F_var = np.squeeze(tf.abs(tf.complex(A2B_var[:,3,:,:,:1],A2B_var[:,3,:,:,1:])),axis=0)
            r2s_var = np.squeeze(A2B_var[:,-1,:,:,1:],axis=0)*(r2_sc**2)
            field_var = np.squeeze(A2B_var[:,-1,:,:,:1],axis=0)*(fm_sc**2)
        
            if test_args.model_sel == 'Mag':
                PDFF_var = WF_var
            else:
                Aux_WF = np.where(f_m_aux>=w_m_aux, f_m_aux, w_m_aux)
                Aux_WF_var = np.where(f_m_aux>=w_m_aux, F_var, W_var)
                PDFF_var = Aux_WF_var/(Aux_WF**2)
                PDFF_var -= 2 * WF_var / (Aux_WF*wf_m_aux)
                PDFF_var += (W_var + F_var + 2*WF_var)/(wf_m_aux)
                PDFF_var *= Aux_WF**2 / (wf_m_aux)**2 #[W_var,WF_var,F_var]
                PDFF_var = np.nan_to_num(PDFF_var)
            
            # Update maps to be saved in MAT file
            PDFF_vars.append(PDFF_var*1e4)
            WF_vars.append(np.expand_dims(np.concatenate([W_var,WF_var,F_var],axis=-1), axis=-2))
            R2_vars.append(r2s_var)
            FM_vars.append(field_var)

    listdir[-1] = listdir[-1] + '_FF.nii.gz'
    outpath = '/'.join(listdir)
    listdir[-1] = listdir[-1].replace("FF","R2s")
    outpath_r2s = '/'.join(listdir)
    if A2B_var is not None:
        listdir[-1] = listdir[-1].replace("R2s","Var")
        outpath_var = '/'.join(listdir)

    affine_matrix = np.eye(4)
    affine_matrix[0,0] = -1.95
    affine_matrix[1,1] = 1.95
    affine_matrix[2,2] = -6.0
    new_origin = np.array([232.115, -201.4852, 130.3964])
    # affine_matrix[:3, 3] = new_origin
    
    F_aux = [np.squeeze(f, axis=-1) for f in F]
    R2_aux = [np.squeeze(r2, axis=-1) for r2 in R2]
    F_nii = np.array(F_aux).transpose([2,1,0])
    R2_nii = np.array(R2_aux).transpose([2,1,0])
    if A2B_var is not None:
        Var_aux = [np.squeeze(f, axis=-1) for f in PDFF_vars]
        Var_nii = np.array(Var_aux).transpose([2,1,0])

    new_pdff = nib.Nifti1Image(np.flip(F_nii,1), affine=affine_matrix)
    # new_pdff = nib.Nifti1Image(F_nii, affine=affine_matrix)
    nib.save(new_pdff, outpath)
    new_r2s = nib.Nifti1Image(np.flip(R2_nii,1), affine=affine_matrix)
    # new_r2s = nib.Nifti1Image(R2_nii, affine=affine_matrix)
    nib.save(new_r2s, outpath_r2s)
    if A2B_var is not None:
        new_var = nib.Nifti1Image(np.flip(Var_nii,1), affine=affine_matrix)
        # new_var = nib.Nifti1Image(Var_nii, affine=affine_matrix)
        nib.save(new_var, outpath_var)

    i += 1