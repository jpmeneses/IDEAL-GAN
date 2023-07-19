import tensorflow as tf
import numpy as np

import DLlib as dl
import pylib as py
import tf2lib as tl
import wflib as wf
import data
from utils import *

import tqdm
import xlsxwriter

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from time import process_time

py.arg('--experiment_dir',default='output/WF-sep')
py.arg('--map',default='PDFF',choices=['PDFF','R2s','Water'])
py.arg('--te_input', type=bool, default=True)
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)

# Excel file for saving ROIs values
workbook = xlsxwriter.Workbook(py.join(args.experiment_dir,args.map+'_phantom_ROIs.xlsx'))

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
dataset_hdf5 = 'phantom_GC_192_complex_2D.hdf5'
testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx,
                                    acqs_data=True, te_data=True,
                                    complex_data=(args.G_model=='complex'),
                                    MEBCRN=(args.G_model=='MEBCRN'))

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,hgt,wdt,n_out = np.shape(testY)

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

# model
if args.G_model == 'multi-decod' or args.G_model == 'encod-decod':
  if args.out_vars == 'WF-PM':
    G_A2B = dl.MDWF_Generator(input_shape=(hgt,wdt,ech_idx),
                              te_input=args.te_input,
                              te_shape=(args.n_echoes,),
                              filters=args.n_G_filters,
                              WF_self_attention=args.D1_SelfAttention,
                              R2_self_attention=args.D2_SelfAttention,
                              FM_self_attention=args.D3_SelfAttention)
  else:
    G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,ech_idx),
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
  G_A2B = dl.UNet(input_shape=(hgt,wdt,ech_idx),
                  n_out=n_out,
                  filters=args.n_G_filters,
                  te_input=args.te_input,
                  te_shape=(args.n_echoes,),
                  self_attention=args.D1_SelfAttention)
elif args.G_model == 'MEBCRN':
  if args.out_vars == 'WFc':
    n_out = 4
    out_activ = None
  else:
    n_out = 2
    out_activ = 'sigmoid'
  G_A2B = dl.MEBCRN(input_shape=(args.n_echoes,hgt,wdt,2),
                    n_outputs=n_out,
                    output_activation=out_activ,
                    filters=args.n_G_filters,
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
    A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
  elif args.out_vars == 'WFc':
    A2B_WF = G_A2B(A, training=False)
    A2B_WF = tf.where(B[:,:,:,:4]!=0,A2B_WF,0.0)
    A2B_WF_real = A2B_WF[:,:,:,0::2]
    A2B_WF_imag = A2B_WF[:,:,:,1::2]
    A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
    A2B_PM = tf.zeros_like(B_PM)
    A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
  elif args.out_vars == 'PM':
    if args.te_input:
      A2B_PM = G_A2B([A,TE], training=False)
    else:
      A2B_PM = G_A2B(A, training=False)
    A2B_PM = tf.where(A[:,:,:,:2]!=0.0,A2B_PM,0.0)
    A2B_R2, A2B_FM = tf.dynamic_partition(A2B_PM,indx_PM,num_partitions=2)
    A2B_R2 = tf.reshape(A2B_R2,B[:,:,:,:1].shape)
    A2B_FM = tf.reshape(A2B_FM,B[:,:,:,:1].shape)
    if args.G_model=='U-Net' or args.G_model=='MEBCRN':
      A2B_FM = (A2B_FM - 0.5) * 2
      A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
      A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
    A2B_WF = wf.get_rho(A,A2B_PM,TE)
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

t1 = process_time()
for A, B, TE in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
  A = tf.expand_dims(A,axis=0)
  B = tf.expand_dims(B,axis=0)
  TE = tf.expand_dims(TE, axis=0)
  A2B = sample(A,B,TE)
  # A2B = tf.expand_dims(A2B,axis=0)

  all_test_ans[i,:,:,:] = A2B
  i += 1

t2 = process_time()
print("Elapsed time during the whole program in seconds:",t2-t1) 
print("Time per slice:",(t2-t1)/len_dataset)

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
tracker = IndexTracker_phantom(fig, ax, X, bool_PDFF, lims, 'phantom_slices_crops.npy')

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('button_press_event', tracker.button_press)
fig.canvas.mpl_connect('key_press_event', tracker.key_press)
plt.show()

# Save slices indexes and crops coordinates
with open('phantom_slices_crops.npy', 'wb') as f:
  np.save(f,np.array(tracker.frms))
  np.save(f,np.array(tracker.crops_1))
  np.save(f,np.array(tracker.crops_2))

GT_vals = [.0,.026,.053,.079,.105,.157,.209,.312,.413,.514,1.0]

for k in range(len_dataset):
  idxs = [i for i,x in enumerate(tracker.frms) if x==k]
  XA_res_all = []
  XA_gt_all = []
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
    XA_res_all.append(XA_res_aux)
    XA_gt_all.append(XA_gt_aux)
  # Export to Excel file
  if len(idxs)>0:
    ws_ROI_1 = workbook.add_worksheet('Slice_'+str(k))
    ws_ROI_1.write(0,0,'Ground-truth')
    ws_ROI_1.write(0,1,'GraphCuts')
    ws_ROI_1.write(0,2,'Model')
    for idx1 in range(len(XA_gt_all)):
      ws_ROI_1.write(idx1+1,0,GT_vals[idx1])
      ws_ROI_1.write(idx1+1,1,XA_gt_all[idx1])
      ws_ROI_1.write(idx1+1,2,XA_res_all[idx1])

workbook.close()