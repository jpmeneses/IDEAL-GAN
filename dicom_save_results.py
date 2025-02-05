import functools

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

import tf2lib as tl
import DLlib as dl
import pylib as py
import wflib as wf
import data

import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir',default='output/WF-IDEAL')
py.arg('--dataset', type=str, default='multiTE', choices=['multiTE','3ech','JGalgani','v33'])
py.arg('--data_size', type=int, default=384, choices=[192,384])
py.arg('--map',default='PDFF',choices=['PDFF','R2s','Water'])
py.arg('--is_GC',type=bool,default=False)
py.arg('--te_input', type=bool, default=False)
py.arg('--ME_layer', type=bool, default=False)
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
test_args = py.args()
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


#################################################################################
############################### SAVE FUNCTION ###################################
#################################################################################

if args.is_GC:
	method_prefix = 'm000'
else:
	method_prefix = 'm' + args.experiment_dir.split('-')[1]

if args.TE1 == 0.0013:
	if args.dTE == 0.0021:
		prot_prefix = 'p00'
	elif args.dTE == 0.0022:
		prot_prefix = 'p01'
	elif args.dTE == 0.0023:
		prot_prefix = 'p02'
	elif args.dTE == 0.0024:
		prot_prefix = 'p03'
elif args.TE1 == 0.0014:
	if args.dTE == 0.0021:
		prot_prefix = 'p04'
	elif args.dTE == 0.0022:
		prot_prefix = 'p05'


#################################################################################
################################# LOAD DATA #####################################
#################################################################################

# data
ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200,300

dataset_dir = '../datasets/'
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
elif args.dataset == '3ech':
  delta_idxs = [21,24,24,24,22,24,25,24,18,21,24,18, 21,24,21,18,16,18,24,21]
  num_slice_list = None
  rnc = False

if args.dataset == 'JGalgani' or args.dataset == '3ech':
  testX, testY=data.load_hdf5(dataset_dir,dataset_hdf5,ech_idx,num_slice_list=None,remove_non_central=rnc,
                              acqs_data=True,te_data=False,remove_zeros=True,MEBCRN=False)
  TEs = np.ones((testX.shape[0],1),dtype=np.float32)
elif args.dataset == 'multiTE':
  testX, testY, TEs =  data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, custom_list=custom_list,
                                      acqs_data=True,te_data=True,remove_zeros=False,MEBCRN=False)
else:
  testX, testY = data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, acqs_data=True, 
                                te_data=False,remove_zeros=True,MEBCRN=False)
  TEs = np.ones((testX.shape[0],1),dtype=np.float32)
if args.dataset == 'multiTE':
  testX, testY, TEs = data.group_TEs(testX,testY,TEs,TE1=args.TE1,dTE=args.dTE,MEBCRN=False)

len_dataset,hgt,wdt,n_out = np.shape(testY)

print('Acquisition Dimensions:', hgt,wdt)
print('Echoes:',args.n_echoes)
print('Output Maps:',n_out)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)

A_B_dataset_test = tf.data.Dataset.from_tensor_slices((testX,testY,TEs))
A_B_dataset_test.batch(1)

# Input and output dimensions (testing data)
print('Testing input shape:',testX.shape)
print('Testing output shape:',testY.shape)


#################################################################################
######################### LOAD MODEL AND GET RESULTS ############################
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
                            te_input=args.te_input,
                            te_shape=(args.n_echoes,),
                            ME_layer=args.ME_layer,
                            filters=args.n_G_filters,
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
if not(args.is_GC):
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
    A2B_WF = wf.get_rho(A,A2B_PM,MEBCRN=False)
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

@tf.function
def sample_GC(B):
  w_all_ans = tf.abs(tf.complex(B[...,:1],B[...,1:2]))
  f_all_ans = tf.abs(tf.complex(B[...,2:3],B[...,3:4]))
  return tf.concat([w_all_ans,f_all_ans,B[...,4:]],axis=-1)

all_test_ans = np.zeros((len_dataset,hgt,wdt,4))
i = 0
for A, B, TE in tqdm.tqdm(A_B_dataset_test, desc='Testing Samples Loop', total=len_dataset):
  A = tf.expand_dims(A,axis=0)
  B = tf.expand_dims(B,axis=0)
  TE= tf.expand_dims(TE,axis=0)
  if args.is_GC:
    A2B = sample_GC(B)
  else:
    A2B = sample(A,B,TE)
  all_test_ans[i,:,:,:] = A2B
  i += 1

w_all_ans = all_test_ans[:,:,:,0]
f_all_ans = all_test_ans[:,:,:,1]
r2_all_ans = all_test_ans[:,:,:,2]*r2_sc

if args.map == 'PDFF':
  PDFF_all_ans = f_all_ans/(w_all_ans+f_all_ans)
  PDFF_all_ans[np.isnan(PDFF_all_ans)] = 0.0
  np.clip(PDFF_all_ans,0,1,out=PDFF_all_ans)
  X = PDFF_all_ans
elif args.map == 'R2':
  X = r2_all_ans
elif args.map == 'Water':
  X = w_all_ans
else:
  raise TypeError('The selected map is not available')

save_dir = py.join(args.experiment_dir, 'out_dicom', args.map)
py.mkdir(save_dir)
pre_filename = args.map + '_' + prot_prefix + '_'
end_filename = '_' + method_prefix

# n_slices = [21,24,24,24,22,24,25,24,18,21,24,18,21,24,21,18,16,18,24,21]
if args.dataset == 'JGalgani':
  n_slices = num_slice_list
  ini_idx = 0
elif args.dataset == 'multiTE':
  n_slices = delta_idxs
  ini_idx = 22
elif args.dataset == '3ech':
  n_slices = delta_idxs
  ini_idx = 24
else:
  n_slices = [21]
  ini_idx = 42
cont = 0
for idx in range(len(n_slices)):
  ini = cont
  fin = cont + n_slices[idx]
  cont = cont + n_slices[idx]

  volun_name = 'v' + str(idx+ini_idx).zfill(3)
  filename = pre_filename + volun_name + end_filename

  # image3d = np.squeeze(out_maps[0:21,:,:,0])
  image3d = np.squeeze(X[ini:fin,:,:])
  image3d = np.moveaxis(image3d,0,-1)

  # Populate required values for file meta information
  ds = data.gen_ds(idx+ini_idx)

  for i in range(0, np.shape(image3d)[2]):
    data.write_dicom(ds, image3d[:,:,i], volun_name, method_prefix, filename, i, np.shape(image3d)[2])