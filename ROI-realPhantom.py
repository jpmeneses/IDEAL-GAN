import tensorflow as tf
import numpy as np

import tf2lib as tl
import DLlib as dl
import pylib as py
import wflib as wf
import data
from utils import *

import tqdm
import xlsxwriter

import matplotlib.pyplot as plt
from time import process_time

py.arg('--experiment_dir',default='TEaug-300')
py.arg('--dataset', type=str, default='phantom_1p5', choices=['phantom_1p5','phantom_3p0'])
py.arg('--model_sel', type=str, default='VET-Net', choices=['U-Net','MDWF-Net','VET-Net','AI-DEAL','GraphCuts'])
py.arg('--remove_ech1', type=bool, default=False)
py.arg('--map',default='PDFF',choices=['PDFF','R2s','Water','PDFF-var'])
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join('output',test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# Excel file for saving ROIs values
workbook = xlsxwriter.Workbook(py.join('output',args.experiment_dir,args.map+'_phantom_ROIs.xlsx'))

r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5 = args.dataset + '_GC_192_128_complex_2D.hdf5'
testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, acqs_data=True,
                                    te_data=True, MEBCRN=True)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,n_out,hgt,wdt,n_ch = np.shape(testY)

print('Acquisition Dimensions:', hgt,wdt)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)
print('TEs shape:',TEs.shape)

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
      A2B_WF = wf.get_rho(A[:,1:,...], A2B_PM, te=TE[:,1:,...])
    else:
      A2B_WF = wf.get_rho(A, A2B_PM, te=TE)
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

t1 = process_time()
for A, B, TE in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
  A = tf.expand_dims(A,axis=0)
  B = tf.expand_dims(B,axis=0)
  TE = tf.expand_dims(TE, axis=0)
  A2B, A2B_var = test(A,B,TE)
  # A2B = tf.expand_dims(A2B,axis=0)

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

t2 = process_time()
print("Elapsed time during the whole program in seconds:",t2-t1) 
print("Time per slice:",(t2-t1)/len_dataset)

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
  PDFF_all_ans = np.where(f_all_ans>=w_all_ans,f_all_ans/wf_all_ans,1-w_all_ans/wf_all_ans)
  PDFF_all_gt = np.where(f_all_gt>=w_all_gt,f_all_gt/wf_all_gt,1-w_all_gt/wf_all_gt)
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
  PDFF_all_ans = np.where(f_all_ans>=w_all_ans,f_all_ans/wf_all_ans,1-w_all_ans/wf_all_ans)
  PDFF_all_gt = np.where(f_all_gt>=w_all_gt,f_all_gt/wf_all_gt,1-w_all_gt/wf_all_gt)
  PDFF_all_ans[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_gt[np.isnan(PDFF_all_gt)] = 0.0
  PDFF_all_ans[np.isnan(PDFF_all_ans)] = 0.0
  X = np.transpose(ffuq_all_ans,(1,2,0))
  X_gt = np.transpose(np.abs(PDFF_all_ans-PDFF_all_gt),(1,2,0))
  lims = (0,1)
else:
  raise TypeError('The selected map is not available')

npy_file = py.join('ROI_files', args.dataset + '_slices_crops.npy')

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker_phantom(fig, ax, X, bool_PDFF, lims, npy_file=npy_file)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('button_press_event', tracker.button_press)
fig.canvas.mpl_connect('key_press_event', tracker.key_press)
plt.show()

# Save slices indexes and crops coordinates
with open(npy_file, 'wb') as f:
  np.save(f,np.array(tracker.frms))
  np.save(f,np.array(tracker.crops_1))
  np.save(f,np.array(tracker.crops_2))

GT_vals = [.0,.026,.053,.079,.105,.157,.209,.312,.413,.514,1.0]

for k in range(len_dataset):
  idxs = [i for i,x in enumerate(tracker.frms) if x==k]
  XA_res_all = []
  XA_gt_all = []
  if args.map == 'PDFF-var':
    XA_q1_all = []
    XA_q2_all = []
    XA_q3_all = []
  for idx in idxs:
    left_x_A = tracker.crops_1[idx][0]
    sup_y_A = tracker.crops_1[idx][1]
    r1_A,r2_A = sup_y_A,(sup_y_A+9)
    c1_A,c2_A = left_x_A,(left_x_A+9)
    XA_all = X[r1_A:r2_A,c1_A:c2_A,k]
    XA_all_gt = X_gt[r1_A:r2_A,c1_A:c2_A,k]
    if args.map == 'PDFF':
      XA_res_aux = np.median(XA_all,axis=(0,1))
      XA_gt_aux = np.median(XA_all_gt,axis=(0,1))
    elif args.map == 'R2s' or args.map == 'Water':
      XA_res_aux = np.mean(XA_all,axis=(0,1))
      XA_gt_aux = np.mean(XA_all_gt,axis=(0,1))
    elif args.map == 'PDFF-var':
      XA_res_aux = np.mean(XA_all,axis=(0,1))
      XA_gt_aux = np.median(XA_all_gt)
      XA_q1_aux = np.quantile(XA_all_gt,0.25)
      XA_q3_aux = np.quantile(XA_all_gt,0.75)
    XA_res_all.append(XA_res_aux)
    XA_gt_all.append(XA_gt_aux)
    if args.map == 'PDFF-var':
      XA_q1_all.append(XA_q1_aux)
      XA_q3_all.append(XA_q3_aux)
  # Export to Excel file
  if len(idxs)>0:
    ws_ROI_1 = workbook.add_worksheet('Slice_'+str(k))
    ws_ROI_1.write(0,0,'Ground-truth')
    if args.map == 'PDFF-var':
      ws_ROI_1.write(0,0,'Q1')
      ws_ROI_1.write(0,1,'Q2')
      ws_ROI_1.write(0,2,'Q3')
      ws_ROI_1.write(0,3,'PDFF Var')
      for idx1 in range(len(XA_gt_all)):
        ws_ROI_1.write(idx1+1,0,XA_q1_all[idx1])
        ws_ROI_1.write(idx1+1,1,XA_gt_all[idx1])
        ws_ROI_1.write(idx1+1,2,XA_q3_all[idx1])
        ws_ROI_1.write(idx1+1,3,XA_res_all[idx1])
    else:
      ws_ROI_1.write(0,1,'Ground-truth')
      ws_ROI_1.write(0,2,'Model res.')
      for idx1 in range(len(XA_gt_all)):
        ws_ROI_1.write(idx1+1,0,GT_vals[idx1])
        ws_ROI_1.write(idx1+1,1,XA_gt_all[idx1])
        ws_ROI_1.write(idx1+1,2,XA_res_all[idx1])

workbook.close()