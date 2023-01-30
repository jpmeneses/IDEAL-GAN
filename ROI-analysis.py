import tensorflow as tf
import numpy as np

import DLlib as dl
import pylib as py
import tf2lib as tl
import wflib as wf
import data
from utils import *

# import statsmodels.api as sm
import tqdm
import xlsxwriter

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import os
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

py.arg('--experiment_dir',default='output/WF-sep')
py.arg('--map',default='PDFF',choices=['PDFF','R2s','Water'])
py.arg('--te_input', type=bool, default=False)
py.arg('--multi_TE', type=bool, default=False)
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# Excel file for saving ROIs values
if args.multi_TE:
  workbook =xlsxwriter.Workbook(py.join(args.experiment_dir,args.map + '_ROIs_'
                                + str(int(np.round(args.TE1*1e4))) + '_' + str(int(np.round(args.dTE*1e4))) 
                                + '.xlsx'))
else:
  workbook = xlsxwriter.Workbook(py.join(args.experiment_dir,args.map+'_ROIs.xlsx'))
ws_ROI_1 = workbook.add_worksheet('RHL')
ws_ROI_1.write(0,0,'Ground-truth')
ws_ROI_1.write(0,1,'Model res.')
ws_ROI_2 = workbook.add_worksheet('LHL')
ws_ROI_2.write(0,0,'Ground-truth')
ws_ROI_2.write(0,1,'Model res.')

# data
ech_idx = args.n_echoes * 2

dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
# dataset_dir = '../MRI-Datasets/'
if args.n_echoes == 6 and not(args.multi_TE):
  dataset_hdf5 = '6ech_GC_192_origTEs_complex_2D.hdf5'
elif args.n_echoes == 6 and args.multi_TE:
  dataset_hdf5 = 'multiTE_GC_192_complex_2D.hdf5'
elif args.n_echoes == 3:
  dataset_hdf5 = '3ech_GC_192_complex_2D.hdf5'
testX, testY, TEs =data.load_hdf5(dataset_dir,dataset_hdf5,ech_idx,acqs_data=True,
                                  te_data=True,complex_data=(args.G_model=='complex'),remove_zeros=False)
if args.multi_TE:
  testX, testY, TEs = data.group_TEs(testX,testY,TEs,TE1=args.TE1,dTE=args.dTE)
  npy_file = 'slices_crops_multiTE.npy'
else:
  npy_file = 'slices_crops_3ech.npy'

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

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY,TEs))
A_B_dataset_test.batch(1)

#################################################################################
################################# LOAD MODEL ####################################
#################################################################################

# model
if args.G_model == 'multi-decod' or args.G_model == 'encod-decod':
  G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,d_ech),
                          filters=args.n_G_filters,
                          te_input=args.te_input,
                          te_shape=(args.n_echoes,),
                          R2_self_attention=args.D1_SelfAttention,
                          FM_self_attention=args.D2_SelfAttention)
elif args.G_model == 'U-Net':
  if args.out_vars == 'WF-PM':
    n_out = 4
  else:
    n_out = 2
  G_A2B = dl.UNet(input_shape=(hgt,wdt,d_ech),
                  n_out=n_out,
                  filters=args.n_G_filters,
                  te_input=args.te_input,
                  te_shape=(args.n_echoes,),
                  self_attention=args.D1_SelfAttention)

# restore
tl.Checkpoint(dict(G_A2B=G_A2B), py.join(args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample(A, B, TE=None):
  indx_B = tf.concat([tf.zeros_like(B[:,:,:,:4],dtype=tf.int32),
                      tf.ones_like(B[:,:,:,4:],dtype=tf.int32)],axis=-1)
  indx_B_abs = tf.concat([tf.zeros_like(B[:,:,:,:2],dtype=tf.int32),
                          tf.ones_like(B[:,:,:,4:],dtype=tf.int32)],axis=-1)
  indx_PM =tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                      tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
  # Split B
  B_WF,B_PM = tf.dynamic_partition(B,indx_B,num_partitions=2)
  B_WF = tf.reshape(B_WF,B[:,:,:,:4].shape)
  B_PM = tf.reshape(B_PM,B[:,:,:,4:].shape)
  # Magnitude of water/fat images
  B_WF_real = B_WF[:,:,:,0::2]
  B_WF_imag = B_WF[:,:,:,1::2]
  B_WF_abs = tf.abs(tf.complex(B_WF_real,B_WF_imag))
  # Split B param maps
  B_R2, B_FM = tf.dynamic_partition(B_PM,indx_PM,num_partitions=2)
  B_R2 = tf.reshape(B_R2,B[:,:,:,:1].shape)
  B_FM = tf.reshape(B_FM,B[:,:,:,:1].shape)
  # Estimate A2B
  if args.out_vars == 'WF':
    if args.te_input:
      A2B_WF_abs = G_A2B([A,TE], training=True)
    else:
      A2B_WF_abs = G_A2B(A, training=True)
    A2B_WF_abs = tf.where(A[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)
    A2B_PM = tf.zeros_like(B_PM)
    # Split A2B param maps
    A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
    A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
    A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
    A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
  elif args.out_vars == 'PM':
    if args.te_input:
      A2B_PM = G_A2B([A,TE], training=True)
    else:
      A2B_PM = G_A2B(A, training=True)
    A2B_PM = tf.where(B_PM!=0.0,A2B_PM,0.0)
    A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
    A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
    A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
    if args.G_model=='U-Net' or args.G_model=='MEBCRN':
      A2B_FM = (A2B_FM - 0.5) * 2
      A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
      A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
    A2B_WF = wf.get_rho(A,A2B_PM)
    A2B_WF_real = A2B_WF[:,:,:,0::2]
    A2B_WF_imag = A2B_WF[:,:,:,1::2]
    A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
    A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
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

  return A2B_abs

all_test_ans = np.zeros((len_dataset,hgt,wdt,4))
i = 0

for A, B, TE in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
  A = tf.expand_dims(A,axis=0)
  B = tf.expand_dims(B,axis=0)
  TE= tf.expand_dims(TE,axis=0)
  A2B = sample(A,B,TE)
  
  all_test_ans[i,:,:,:] = A2B
  i += 1

w_all_ans = all_test_ans[:,:,:,0]
f_all_ans = all_test_ans[:,:,:,1]
r2_all_ans = all_test_ans[:,:,:,2]*r2_sc

# Ground truth
w_all_gt = np.abs(tf.complex(testY[:,:,:,0],testY[:,:,:,1]))
f_all_gt = np.abs(tf.complex(testY[:,:,:,2],testY[:,:,:,3]))
r2_all_gt = testY[:,:,:,4]*r2_sc

#################################################################################
#################################################################################

if args.map == 'PDFF':
  bool_PDFF = True
  PDFF_all_ans = f_all_ans/(w_all_ans+f_all_ans)
  PDFF_all_gt = f_all_gt/(w_all_gt+f_all_gt)
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
else:
  raise TypeError('The selected map is not available')

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(fig, ax, X, bool_PDFF, lims, npy_file=npy_file)

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
  for idx in range(len(tracker.frms)):
    k = tracker.frms[idx]
    # Crop A
    left_x_A = tracker.crops_1[idx][0]
    sup_y_A = tracker.crops_1[idx][1]
    r1_A,r2_A = sup_y_A,(sup_y_A+9)
    c1_A,c2_A = left_x_A,(left_x_A+9)
    XA_all = X[r1_A:r2_A,c1_A:c2_A,k]
    XA_all_gt = X_gt[r1_A:r2_A,c1_A:c2_A,k]
    # Crop B
    left_x_B = tracker.crops_2[idx][0]
    sup_y_B = tracker.crops_2[idx][1]
    r1_B,r2_B = sup_y_B,(sup_y_B+9)
    c1_B,c2_B = left_x_B,(left_x_B+9)
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
    # Crop A
    XA_res_all.append(XA_res_aux)
    XA_gt_all.append(XA_gt_aux)
    # Crop B
    XB_res_all.append(XB_res_aux)
    XB_gt_all.append(XB_gt_aux)
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
  for idx1 in range(len(XA_gt_all)):
    ws_ROI_1.write(idx1+1,0,XA_gt_all[idx1])
    ws_ROI_1.write(idx1+1,1,XA_res_all[idx1])
  for idx2 in range(len(XB_gt_all)):
    ws_ROI_2.write(idx2+1,0,XB_gt_all[idx2])
    ws_ROI_2.write(idx2+1,1,XB_res_all[idx2])

workbook.close()