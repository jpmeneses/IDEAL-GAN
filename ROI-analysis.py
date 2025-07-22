import sys
import numpy as np
import tensorflow as tf

# OPTIONAL - DISABLE GPU
tf.config.experimental.set_visible_devices([], 'GPU')

import tf2lib as tl
import DLlib as dl
import pylib as py
import wflib as wf
import data

from utils import *

import matplotlib.pyplot as plt
import tqdm
import xlsxwriter
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='TEaug-300')
py.arg('--dataset', type=str, default='multiTE', choices=['multiTE','3ech','JGalgani','Attilio'])
py.arg('--data_size', type=int, default=384, choices=[192,384])
py.arg('--model_sel', type=str, default='VET-Net', choices=['U-Net','MDWF-Net','VET-Net','AI-DEAL','GraphCuts'])
py.arg('--remove_ech1', type=bool, default=False)
py.arg('--phase_constraint', type=bool, default=False)
py.arg('--magnitude_disc', type=bool, default=False)
py.arg('--map',default='PDFF',choices=['PDFF','R2s','Water','PDFF-var'])
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
py.arg('--batch_size', type=int, default=1)
py.arg('--display', type=bool, default=True)
test_args = py.args()
args = py.args_from_yaml(py.join('output', test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

if not(hasattr(args,'field')):
  py.arg('--field', type=float, default=1.5)
  ds_args = py.args()
  args.__dict__.update(ds_args.__dict__)

if not(hasattr(args,'n_echoes')):
  py.arg('--n_echoes', type=int, default=6)
  ne_args = py.args()
  args.__dict__.update(ne_args.__dict__)

# Excel file for saving ROIs values
if args.dataset == 'multiTE':
  out_filename = args.map + '_ROIs_' + str(int(np.round(args.TE1*1e4))) + '_' + str(int(np.round(args.dTE*1e4)))
else:
  out_filename = args.map+'_'+args.dataset+'_ROIs'
if args.phase_constraint:
  out_filename += '_pc'
if args.magnitude_disc:
  out_filename += '_md'
workbook = xlsxwriter.Workbook(py.join('output',args.experiment_dir,out_filename + '.xlsx'))
ws_ROI_1 = workbook.add_worksheet('RHL')
ws_ROI_2 = workbook.add_worksheet('LHL')
if args.map == 'PDFF-var':
  ws_ROI_1.write(0,0,'Q1')
  ws_ROI_1.write(0,1,'Q2')
  ws_ROI_1.write(0,2,'Q3')
  ws_ROI_1.write(0,3,'PDFF Var')
  ws_ROI_2.write(0,0,'Q1')
  ws_ROI_2.write(0,1,'Q2')
  ws_ROI_2.write(0,2,'Q3')
  ws_ROI_2.write(0,3,'PDFF Var')
else:
  ws_ROI_1.write(0,0,'Ground-truth')
  ws_ROI_1.write(0,1,'Model res.')
  ws_ROI_2.write(0,0,'Ground-truth')
  ws_ROI_2.write(0,1,'Model res.')

# data
if args.n_echoes>0:
  ech_idx = args.n_echoes * 2
else:
  ech_idx = 12

dataset_dir = '../datasets/'
dataset_hdf5 = args.dataset + '_GC_' + str(args.data_size) + '_complex_2D.hdf5'
npy_file = py.join('ROI_files', 'slices_crops_' + str(args.dataset) + '_' + str(args.data_size) + '.npy')
if args.dataset == 'JGalgani':
  num_slice_list = [21,20,24,24,21,24,21,21,24,21,21,22,27,23,22,20,24,21,21,22,20]
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

if args.dataset == 'JGalgani' or args.dataset == '3ech' or args.dataset == 'Attilio':
  testX, testY = data.load_hdf5(dataset_dir,dataset_hdf5,ech_idx,num_slice_list=num_slice_list,remove_non_central=rnc,
                                acqs_data=True,te_data=False,remove_zeros=True,MEBCRN=True)
  TEs = np.ones((testX.shape[0],1),dtype=np.float32)
else:
  testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, custom_list=custom_list,
                                      acqs_data=True,te_data=True,remove_zeros=False,MEBCRN=True)
  print('Testing input shape before TE grouping:',testX.shape)
  testX, testY, TEs = data.group_TEs(testX,testY,TEs,TE1=args.TE1,dTE=args.dTE,MEBCRN=True)

len_dataset,n_out,hgt,wdt,n_ch = np.shape(testY)
r2_sc,fm_sc = 200,300

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',args.n_echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY,TEs))
A_B_dataset_test.batch(1)

#################################################################################
################################# LOAD MODEL ####################################
#################################################################################

if args.model_sel == 'U-Net':
  G_A2B = dl.UNet(input_shape=(None,None,2*args.n_echoes), n_out=2, filters=args.n_G_filters,output_activation='relu')
  checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join('output', args.experiment_dir, 'checkpoints'))
elif args.model_sel == 'MDWF-Net':
  G_A2B = dl.MDWF_Generator(input_shape=(None,None,2*args.n_echoes), filters=args.n_G_filters,
                            WF_self_attention=args.D1_SelfAttention, R2_self_attention=args.D2_SelfAttention,
                            FM_self_attention=args.D3_SelfAttention)
  checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join('output', args.experiment_dir, 'checkpoints'))
elif args.model_sel == 'VET-Net':
  G_A2B = dl.PM_Generator(input_shape=(None,None,None,2), te_input=True, te_shape=(None,), filters=args.n_G_filters)
  checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B), py.join('output', args.experiment_dir, 'checkpoints'))
elif args.model_sel == 'AI-DEAL':
  G_A2B = dl.UNet(input_shape=(None,None,None,2), bayesian=True, ME_layer=True, filters=args.n_G_filters,
                  self_attention=args.D1_SelfAttention)
  G_A2R2= dl.UNet(input_shape=(None,None,None,1), bayesian=True, ME_layer=True, filters=args.n_G_filters,
                  output_activation='sigmoid', self_attention=args.D2_SelfAttention)
  checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B, G_A2R2=G_A2R2), py.join('output', args.experiment_dir, 'checkpoints'))

try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

@tf.function
def sample(A, B, TE=None):
  if args.model_sel == 'U-Net':
    A_pf = data.A_from_MEBCRN(A) # CHANGE TO NON-MEBCRN FORMAT
    A2B_WF_abs = G_A2B(A_pf, training=False)
    A2B_WF_abs = tf.expand_dims(A2B_WF_abs, axis=1)
    A2B_WF_abs = tf.transpose(A2B_WF_abs, perm=[0,4,2,3,1])
    A2B_WF = tf.concat([A2B_WF_abs, tf.zeros_like(A2B_WF_abs)], axis=-1)
    A2B = tf.concat([A2B_WF, tf.zeros_like(A2B_WF[:,:1,...])], axis=1)
    A2B = tf.where(A[:,:1,...]!=0.0, A2B, 0.0)
    A2B_var = None
  elif args.model_sel == 'MDWF-Net':
    A_pf = data.A_from_MEBCRN(A) # CHANGE TO NON-MEBCRN FORMAT
    A2B = G_A2B(A_pf, training=False)
    A2B = tf.expand_dims(A2B, axis=1)
    A2B_PM = A2B[...,-1:-3:-1]
    A2B_WF_abs = A2B[...,:2]
    A2B_WF_abs = tf.transpose(A2B_WF_abs, perm=[0,4,2,3,1])
    A2B_WF = tf.concat([A2B_WF_abs, tf.zeros_like(A2B_WF_abs)], axis=-1)
    A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
    A2B = tf.where(A[:,:1,...]!=0.0, A2B, 0.0)
    A2B_var = None
  elif args.model_sel == 'VET-Net':
    if TE is None:
      TE = wf.gen_TEvar(A.shape[1], bs=A.shape[0], orig=True)
    A2B_PM = G_A2B([A,TE], training=False) #[:,:ech_sel.value,...]
    A2B_PM = tf.where(A[:,:1,...]!=0.0, A2B_PM, 0.0)
    if args.remove_ech1:
      A2B_WF = wf.get_rho(A[:,1:,...], A2B_PM, te=TE[:,1:,...], phase_constraint=args.phase_constraint)
    else:
      A2B_WF = wf.get_rho(A, A2B_PM, te=TE, phase_constraint=args.phase_constraint)
    A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
    A2B_var = None
  elif args.model_sel == 'AI-DEAL':
    A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A), axis=-1, keepdims=True))
    if args.remove_ech1:
      A2B_FM = G_A2B(A[:,1:,...], training=False)
    else:
      A2B_FM = G_A2B(A, training=False)
    A2B_R2 = G_A2R2(A_abs, training=False)
    A2B_PM = tf.concat([A2B_FM.mean(),A2B_R2.mean()],axis=-1)
    if args.remove_ech1:
      if TE is None:
        A2B_WF, A2B_WF_var = wf.PDFF_uncertainty(A[:,1:,...], A2B_FM, A2B_R2, rem_R2=False)
      else:
        A2B_WF, A2B_WF_var = wf.PDFF_uncertainty(A[:,1:,...], A2B_FM, A2B_R2, te=TE[:,1:,...], rem_R2=False)
    else:
      A2B_WF, A2B_WF_var = wf.PDFF_uncertainty(A, A2B_FM, A2B_R2, te=TE, rem_R2=False)
    A2B_WF_var = tf.concat([A2B_WF_var,tf.zeros_like(A2B_WF_var)],axis=-1)
    A2B_PM_var = tf.concat([A2B_FM.variance(),A2B_R2.variance()],axis=-1)
    A2B_var = tf.concat([A2B_WF_var,A2B_PM_var], axis=1)
    if A.shape[1] >= A2B_var.shape[1]:
      A2B_var = tf.where(A[:,:5,...]!=0.0, A2B_var, 1e-10)
    else:
      A_aux = tf.concat([A,A],axis=1)
      A2B_var = tf.where(A_aux[:,:5,...]!=0.0, A2B_var, 1e-10)
    A2B = tf.concat([A2B_WF, A2B_PM], axis=1)
    A2B = tf.where(A[:,:3,...]!=0.0, A2B, 0.0)
  return A2B, A2B_var

def test(A, B, TE=None):
  A2B, A2B_var = sample(A, B, TE)
  return A2B, A2B_var

if args.map == 'PDFF-var':
  all_test_ans = np.zeros((len_dataset,hgt,wdt,5))
else:
  all_test_ans = np.zeros((len_dataset,hgt,wdt,4))
i = 0

for A, B, TE in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
  A = tf.expand_dims(A,axis=0)
  B = tf.expand_dims(B,axis=0)
  TE= tf.expand_dims(TE,axis=0)
  if args.model_sel == 'GraphCuts':
    A2B = B
  elif args.dataset == 'JGalgani' or args.dataset == '3ech':
    A2B, A2B_var = test(A,B)
  else:
    A2B, A2B_var = test(A,B,TE)

  A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B[:,:2,:,:,:]),axis=-1))
  A2B_WF_abs = tf.transpose(A2B_WF_abs,perm=[0,2,3,1])
  A2B_WFsum_abs = tf.math.sqrt(tf.reduce_sum(tf.square(tf.reduce_sum(A2B[:,:2,:,:,:], axis=1, keepdims=True)),axis=-1))
  A2B_WFsum_abs = tf.transpose(A2B_WFsum_abs,perm=[0,2,3,1])
  A2B_R2 = A2B[:,2,:,:,1:]
  A2B = tf.concat([A2B_WF_abs,A2B_WFsum_abs,A2B_R2],axis=-1)

  if args.map == 'PDFF-var':
    W_var = tf.abs(tf.complex(A2B_var[:,0,:,:,:1],A2B_var[:,0,:,:,1:]))
    WF_var = tf.abs(tf.complex(A2B_var[:,1,:,:,:1],A2B_var[:,1,:,:,1:]))
    F_var = tf.abs(tf.complex(A2B_var[:,3,:,:,:1],A2B_var[:,3,:,:,1:]))
    r2s_var = A2B_var[:,-1,:,:,1:]*(r2_sc**2)

    PDFF_var = W_var/(A2B_WF_abs[...,:1]**2)
    PDFF_var -= 2 * WF_var / (A2B_WF_abs[...,:1]*A2B_WFsum_abs)
    PDFF_var += (W_var + F_var + 2*WF_var)/(A2B_WF_abs[...,:1])
    PDFF_var *= A2B_WF_abs[...,:1]**2 / (A2B_WFsum_abs)**2 #[W_var,WF_var,F_var]

    A2B = tf.concat([A2B,PDFF_var],axis=-1)

  all_test_ans[i,:,:,:] = A2B
  i += 1

w_all_ans = all_test_ans[:,:,:,0]
f_all_ans = all_test_ans[:,:,:,1]
wf_all_ans = all_test_ans[:,:,:,2]
r2_all_ans = all_test_ans[:,:,:,3]*r2_sc
if args.map == 'PDFF-var':
  ffuq_all_ans = all_test_ans[:,:,:,4]

# Ground truth
w_all_gt = np.sqrt(np.sum(testY[:,0,:,:,:]**2,axis=-1))
f_all_gt = np.sqrt(np.sum(testY[:,1,:,:,:]**2,axis=-1))
wf_all_gt = np.sqrt(np.sum((testY[:,0,...]+testY[:,1,...])**2,axis=-1))
r2_all_gt = testY[:,2,:,:,1]*r2_sc


#################################################################################
#################################################################################

if args.map == 'PDFF':
  bool_PDFF = True
  if args.magnitude_disc:
    PDFF_all_ans = np.where(f_all_ans>=w_all_ans,f_all_ans/wf_all_ans,1-w_all_ans/wf_all_ans)
    PDFF_all_gt = np.where(f_all_gt>=w_all_gt,f_all_gt/wf_all_gt,1-w_all_gt/wf_all_gt)
  else:
    PDFF_all_ans = f_all_ans/wf_all_ans
    PDFF_all_gt = f_all_gt/wf_all_gt
  PDFF_all_ans[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_gt[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_ans[np.isnan(PDFF_all_ans)] = 0.0
  X = np.transpose(PDFF_all_ans,(1,2,0))
  X_gt = np.transpose(PDFF_all_gt,(1,2,0))
  lims = (0,1)
elif args.map == 'R2s':
  bool_PDFF = False
  X = np.transpose(r2_all_ans,(1,2,0))
  X_gt = np.transpose(r2_all_gt,(1,2,0))
  lims = (0,r2_sc)
elif args.map == 'Water':
  bool_PDFF = True
  X = np.transpose(w_all_ans,(1,2,0))
  X_gt = np.transpose(w_all_gt,(1,2,0))
  lims = (0,1)
elif args.map == 'PDFF-var':
  bool_PDFF = True
  ffuq_all_ans[np.isnan(ffuq_all_ans)] = 0.0
  if args.magnitude_disc:
    PDFF_all_ans = np.where(f_all_ans>=w_all_ans,f_all_ans/wf_all_ans,1-w_all_ans/wf_all_ans)
    PDFF_all_gt = np.where(f_all_gt>=w_all_gt,f_all_gt/wf_all_gt,1-w_all_gt/wf_all_gt)
  else:
    PDFF_all_ans = f_all_ans/wf_all_ans
    PDFF_all_gt = f_all_gt/wf_all_gt
  PDFF_all_ans[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_gt[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_ans[np.isnan(PDFF_all_ans)] = 0.0
  X = np.transpose(ffuq_all_ans,(1,2,0))
  X_gt = np.transpose(PDFF_all_gt,(1,2,0))
  X_ans = np.transpose(PDFF_all_ans,(1,2,0))
  lims = (0,1)
else:
  raise TypeError('The selected map is not available')

if args.data_size == 192:
  ROI_wdt = 8
elif args.data_size == 384:
  ROI_wdt = 16

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(fig, ax, X, bool_PDFF, lims, wdt=ROI_wdt, npy_file=npy_file)

if args.display:
  fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
  fig.canvas.mpl_connect('button_press_event', tracker.button_press)
  fig.canvas.mpl_connect('key_press_event', tracker.key_press)
  plt.show()

# Save slices indexes and crops coordinates
with open(npy_file, 'wb') as f:
  np.save(f,np.array(tracker.frms))
  np.save(f,np.array(tracker.crops_1))
  np.save(f,np.array(tracker.crops_2))

if args.map != 'Water':
  XA_res_all = []
  XA_gt_all = []
  XB_res_all = []
  XB_gt_all = []
  if args.map == 'PDFF-var':
    XA_q1_all = []
    XA_q2_all = []
    XA_q3_all = []
    XB_q1_all = []
    XB_q2_all = []
    XB_q3_all = []
  for idx in range(len(tracker.frms)):
    k = tracker.frms[idx]
    # Crop A
    left_x_A = tracker.crops_1[idx][0]
    sup_y_A = tracker.crops_1[idx][1]
    r1_A,r2_A = sup_y_A,(sup_y_A+ROI_wdt+1)
    c1_A,c2_A = left_x_A,(left_x_A+ROI_wdt+1)
    XA_all = X[r1_A:r2_A,c1_A:c2_A,k]
    XA_all_gt = X_gt[r1_A:r2_A,c1_A:c2_A,k]
    # Crop B
    left_x_B = tracker.crops_2[idx][0]
    sup_y_B = tracker.crops_2[idx][1]
    r1_B,r2_B = sup_y_B,(sup_y_B+ROI_wdt+1)
    c1_B,c2_B = left_x_B,(left_x_B+ROI_wdt+1)
    XB_all = X[r1_B:r2_B,c1_B:c2_B,k]
    XB_all_gt = X_gt[r1_B:r2_B,c1_B:c2_B,k]
    if args.map == 'PDFF':
      # Crop A
      XA_res_aux = np.median(XA_all,axis=(0,1))
      XA_gt_aux = np.median(XA_all_gt,axis=(0,1))
      # Crop B
      XB_res_aux = np.median(XB_all,axis=(0,1))
      XB_gt_aux = np.median(XB_all_gt,axis=(0,1))
    elif args.map == 'R2s' or args.map == 'Water':
      # Crop A
      XA_res_aux = np.mean(XA_all,axis=(0,1))
      XA_gt_aux = np.mean(XA_all_gt,axis=(0,1))
      # Crop B
      XB_res_aux = np.mean(XB_all,axis=(0,1))
      XB_gt_aux = np.mean(XB_all_gt,axis=(0,1))
    elif args.map == 'PDFF-var':
      XA_all_ans = X_ans[r1_A:r2_A,c1_A:c2_A,k]
      XB_all_ans = X_ans[r1_B:r2_B,c1_B:c2_B,k]
      # Crop A
      XA_res_aux = np.mean(XA_all,axis=(0,1))
      XA_gt_aux = np.median(np.abs(XA_all_ans-np.median(XA_all_gt)))
      XA_q1_aux = np.quantile(np.abs(XA_all_ans-np.median(XA_all_gt)),0.25)
      XA_q3_aux = np.quantile(np.abs(XA_all_ans-np.median(XA_all_gt)),0.75)
      # Crop B
      XB_res_aux = np.mean(XB_all,axis=(0,1))
      XB_gt_aux = np.median(np.abs(XB_all_ans-np.median(XB_all_gt)))
      XB_q1_aux = np.quantile(np.abs(XB_all_ans-np.median(XB_all_gt)),0.25)
      XB_q3_aux = np.quantile(np.abs(XB_all_ans-np.median(XB_all_gt)),0.75)
    # Crop A
    XA_res_all.append(XA_res_aux)
    XA_gt_all.append(XA_gt_aux)
    # Crop B
    XB_res_all.append(XB_res_aux)
    XB_gt_all.append(XB_gt_aux)
    if args.map == 'PDFF-var':
      XA_q1_all.append(XA_q1_aux)
      XA_q3_all.append(XA_q3_aux)
      XB_q1_all.append(XB_q1_aux)
      XB_q3_all.append(XB_q3_aux)
  print('Max. measurements:\n',
        np.max(np.abs(XA_res_all)),
        np.max(np.abs(XB_res_all))
        )
  XA_err_list = np.array(XA_res_all)-np.array(XA_gt_all)
  XB_err_list = np.array(XB_res_all)-np.array(XB_gt_all)
  # Histograms of differences
  f,(ax1,ax2) = plt.subplots(figsize=(8,7),nrows=2,ncols=1)
  if args.map == 'PDFF':
    bins = np.linspace(-0.03,0.03,25)
    # First ROI
    N_0,bins_0,patches_0 = ax1.hist(XA_err_list,bins=bins,density=False)
    fracs_0 = N_0/N_0.max()
    norm_0 = colors.Normalize(fracs_0.min(), fracs_0.max())
    for thisfrac_0, thispatch_0 in zip(fracs_0, patches_0):
      color_0 = plt.cm.viridis(norm_0(thisfrac_0))
      thispatch_0.set_facecolor(color_0)
    ax1.set_xlim([-0.03,0.03])
    ax1.set_xlabel('Right Posterior Hepatic Lobe PDFF [%]')
    ax1.set_ylabel('Samples')
    print(np.max(XA_err_list),len(XA_err_list))
    # Second ROI
    N_1,bins_1,patches_1 = ax2.hist(XB_err_list,bins=bins,density=False)
    fracs_1 = N_1/N_1.max()
    norm_1 = colors.Normalize(fracs_1.min(), fracs_1.max())
    for thisfrac_1, thispatch_1 in zip(fracs_1, patches_1):
      color_1 = plt.cm.viridis(norm_1(thisfrac_1))
      thispatch_1.set_facecolor(color_1)
    ax2.set_xlim([-0.03,0.03])
    ax2.set_xlabel('Left Hepatic Lobe PDFF [%]')
    ax2.set_ylabel('Samples')
    print(np.max(XB_err_list),len(XB_err_list))
  elif args.map == 'R2s':
    bins = np.linspace(-10,10,41)
    # First ROI
    N_0,bins_0,patches_0 = ax1.hist(XA_err_list,bins=bins,density=False)
    fracs_0 = N_0/N_0.max()
    norm_0 = colors.Normalize(fracs_0.min(), fracs_0.max())
    for thisfrac_0, thispatch_0 in zip(fracs_0, patches_0):
      color_0 = plt.cm.viridis(norm_0(thisfrac_0))
      thispatch_0.set_facecolor(color_0)
    ax1.set_xlim([-10,10])
    ax1.set_xlabel('Right Posterior Hepatic Lobe R2* [1/s]')
    ax1.set_ylabel('Samples')
    print(np.max(XA_res_all),len(XA_res_all))
    # Second ROI
    N_1,bins_1,patches_1 = ax2.hist(XB_err_list,bins=bins,density=False)
    fracs_1 = N_1/N_1.max()
    norm_1 = colors.Normalize(fracs_1.min(), fracs_1.max())
    for thisfrac_1, thispatch_1 in zip(fracs_1, patches_1):
      color_1 = plt.cm.viridis(norm_1(thisfrac_1))
      thispatch_1.set_facecolor(color_1)
    ax2.set_xlim([-10,10])
    ax2.set_xlabel('Left Hepatic Lobe R2* [1/s]')
    ax2.set_ylabel('Samples')
    print(np.max(XB_res_all),len(XB_res_all))
  # plt.savefig('out_images/ROI_histogram_R2.eps',format='eps')
  if args.display:
    plt.show()
  # Bland-Altman
  # f,(ax1,ax2) = plt.subplots(figsize=(8,7),nrows=2,ncols=1)
  # sm.graphics.mean_diff_plot(np.array(XA_res_all),np.array(XA_gt_all),ax=ax1)
  # if args.map == 'PDFF':
  #   ax1.set_xlim([0,0.45])
  #   ax1.set_ylim([-0.035,0.035])
  # elif args.map == 'R2s':
  #   ax1.set_xlim([20,70])
  #   ax1.set_ylim([-20,20])
  # sm.graphics.mean_diff_plot(np.array(XB_res_all),np.array(XB_gt_all),ax=ax2)
  # if args.map == 'PDFF':
  #   ax2.set_xlim([0,0.45])
  #   ax2.set_ylim([-0.035,0.035])
  # elif args.map == 'R2s':
  #   ax2.set_xlim([20,70])
  #   ax2.set_ylim([-20,20])
  # plt.show()
  # Export to Excel file
  if args.map == 'PDFF-var':
    for idx1 in range(len(XA_gt_all)):
      ws_ROI_1.write(idx1+1,0,XA_q1_all[idx1])
      ws_ROI_1.write(idx1+1,1,XA_gt_all[idx1])
      ws_ROI_1.write(idx1+1,2,XA_q3_all[idx1])
      ws_ROI_1.write(idx1+1,3,XA_res_all[idx1])
    for idx2 in range(len(XB_gt_all)):
      ws_ROI_2.write(idx2+1,0,XB_q1_all[idx1])
      ws_ROI_2.write(idx2+1,1,XB_gt_all[idx2])
      ws_ROI_2.write(idx2+1,0,XB_q3_all[idx1])
      ws_ROI_2.write(idx2+1,3,XB_res_all[idx2])
  else:
    for idx1 in range(len(XA_gt_all)):
      ws_ROI_1.write(idx1+1,0,XA_gt_all[idx1])
      ws_ROI_1.write(idx1+1,1,XA_res_all[idx1])
    for idx2 in range(len(XB_gt_all)):
      ws_ROI_2.write(idx2+1,0,XB_gt_all[idx2])
      ws_ROI_2.write(idx2+1,1,XB_res_all[idx2])

workbook.close()