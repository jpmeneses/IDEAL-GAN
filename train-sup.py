import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tf2lib as tl
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf

import data

from itertools import cycle

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='WF-sup')
py.arg('--data_size', type=int, default=192, choices=[192,384])
py.arg('--DL_gen', type=bool, default=False)
py.arg('--DL_partial_real', type=int, default=0, choices=[0,2,6,10])
py.arg('--DL_filename', default='LDM_ds')
py.arg('--sigma_noise', type=float, default=0.0)
py.arg('--shuffle', type=bool, default=True)
py.arg('--n_echoes', type=int, default=6)
py.arg('--TE1', type=float, default=0.0013)
py.arg('--dTE', type=float, default=0.0021)
py.arg('--field', type=float, default=1.5)
py.arg('--out_vars', default='WF', choices=['WF','WFc','PM','WF-PM'])
py.arg('--G_model', default='multi-decod', choices=['multi-decod','U-Net','MEBCRN'])
py.arg('--n_G_filters', type=int, default=72)
py.arg('--batch_size', type=int, default=8)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0005)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.9999)
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--FM_TV_weight', type=float, default=0.0)
py.arg('--R2_L1_weight', type=float, default=0.0)
py.arg('--FM_L1_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=False)
py.arg('--D2_SelfAttention',type=bool, default=True)
py.arg('--D3_SelfAttention',type=bool, default=True)
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

ech_idx = args.n_echoes * 2
r2_sc,fm_sc = 200.0,300.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'

dataset_hdf5_1 = 'INTA_GC_' + str(args.data_size) + '_complex_2D.hdf5'
valX, valY = data.load_hdf5(dataset_dir, dataset_hdf5_1, ech_idx,
                            acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX,valY))
A_B_dataset_val.batch(1)

if not(args.DL_gen):
    dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

    dataset_hdf5_3 = 'Volunteers_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_3, out_maps_3 = data.load_hdf5(dataset_dir, dataset_hdf5_3, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

    dataset_hdf5_4 = 'Attilio_GC_' + str(args.data_size) + '_complex_2D.hdf5'
    acqs_4, out_maps_4 = data.load_hdf5(dataset_dir, dataset_hdf5_4, ech_idx,
                                        acqs_data=True, te_data=False, MEBCRN=(args.G_model=='MEBCRN'))

    trainX  = np.concatenate((acqs_2,acqs_3,acqs_4),axis=0)
    trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4),axis=0)

    if args.G_model == 'MEBCRN':
        len_dataset,_,_,_,_ = np.shape(trainY)
        _,n_out,hgt,wdt,n_ch = np.shape(valY)
    else:
        len_dataset,_,_,_ = np.shape(trainY)
        _,hgt,wdt,n_out = np.shape(valY)
        n_ch = 2

    A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX,trainY))

else:
    recordPath = py.join('tfrecord', args.DL_filename)
    tfr_dataset = tf.data.TFRecordDataset([recordPath])
    # Create a description of the features.
    feature_description = {
        'acqs': tf.io.FixedLenFeature([], tf.string),
        'out_maps': tf.io.FixedLenFeature([], tf.string),
        }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        parsed_ds = tf.io.parse_example(example_proto, feature_description)
        return tf.io.parse_tensor(parsed_ds['acqs'], out_type=tf.float32), tf.io.parse_tensor(parsed_ds['out_maps'], out_type=tf.float32)

    if args.DL_partial_real != 0:
        if args.TE1 == 0.0014 and args.dTE == 0.0022:
            dataset_hdf5_1 = 'multiTE_' + str(args.data_size) + '_complex_2D.hdf5'
            ini_idxs = [0,84,204,300,396,484,580,680,776,848]#,932,1028, 1100,1142,1190,1232,1286,1334,1388,1460]
            delta_idxs = [21,24,24,24,22,24,25,24,18]#,21,24,18, 21,24,21,18,16,18,24,21]
            end_idx = np.sum(delta_idxs)
            k_idxs = [(0,1),(2,3)]
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
                trainX, trainY, TEs =data.load_hdf5(dataset_dir, dataset_hdf5, ech_idx, custom_list=custom_list,
                                                    acqs_data=True,te_data=True,remove_zeros=False,
                                                    MEBCRN=True, mag_and_phase=True, unwrap=True)
        else:
            if args.DL_partial_real == 2:
                end_idx = 62
            elif args.DL_partial_real == 6:
                end_idx = 200
            elif args.DL_partial_real == 10:
                end_idx = 330
            dataset_hdf5_2 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
            trainX, trainY = data.load_hdf5(dataset_dir,dataset_hdf5_2, ech_idx, end=end_idx,
                                            acqs_data=True, te_data=False, MEBCRN=True,
                                            mag_and_phase=True, unwrap=True)
        A_B_dataset = tfr_dataset.skip(end_idx).map(_parse_function)
        A_B_dataset_aux = tf.data.Dataset.from_tensor_slices((trainX,trainY))
        A_B_dataset = A_B_dataset.concatenate(A_B_dataset_aux)
    else:
        A_B_dataset = tfr_dataset.map(_parse_function)

    for A, B in A_B_dataset.take(1):
        _,hgt,wdt,_ = B.shape
    len_dataset = int(args.DL_filename.split('_')[-1])
    if args.DL_partial_real != 0:
        len_dataset += trainX.shape[0]

A_B_dataset = A_B_dataset.batch(args.batch_size)
if args.shuffle:
    A_B_dataset = A_B_dataset.shuffle(len_dataset)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

if args.G_model == 'multi-decod':
    if args.out_vars == 'WF-PM':
        G_A2B=dl.MDWF_Generator(input_shape=(hgt,wdt,args.n_echoes*2),
                                filters=args.n_G_filters,
                                WF_self_attention=args.D1_SelfAttention,
                                R2_self_attention=args.D2_SelfAttention,
                                FM_self_attention=args.D3_SelfAttention)
    else:
        G_A2B = dl.PM_Generator(input_shape=(hgt,wdt,args.n_echoes*2),
                                ME_layer=False,
                                filters=args.n_G_filters,
                                R2_self_attention=args.D1_SelfAttention,
                                FM_self_attention=args.D2_SelfAttention)

elif args.G_model == 'U-Net':
    if args.out_vars == 'WFc':
        n_out = 4
        out_activ = 'tanh'
    elif args.out_vars == 'WF-PM':
        n_out = 4
        out_activ = 'relu'
    else:
        n_out = 2
        out_activ = 'relu'
    G_A2B = dl.UNet(input_shape=(hgt,wdt,args.n_echoes*2),
                    n_out=n_out,
                    filters=args.n_G_filters,
                    output_activation=out_activ,
                    self_attention=args.D1_SelfAttention)
    if args.out_vars != 'WF':
        trainY[:,:,:,-1]    = 0.5*trainY[:,:,:,-1] + 0.5
        valY[:,:,:,-1]      = 0.5*valY[:,:,:,-1] + 0.5

elif args.G_model == 'MEBCRN':
    if args.out_vars=='WFc':
        n_out = 4
        out_activ = None
    else:
        n_out = 2
        out_activ = 'sigmoid'
    G_A2B=dl.MEBCRN(input_shape=(args.n_echoes,hgt,wdt,2),
                    n_outputs=n_out,
                    output_activation=out_activ,
                    n_res_blocks=9,
                    n_downsamplings=0,
                    filters=args.n_G_filters,
                    self_attention=args.D1_SelfAttention)

else:
    raise(NameError('Unrecognized Generator Architecture'))

IDEAL_op = wf.IDEAL_Layer(field=args.field)

sup_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = dl.LinearDecay(args.lr, total_steps, args.epoch_decay * total_steps / args.epochs)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B, te=None):
    if (args.TE1 != 0.0013) and (args.dTE != 0.0021):
        A = IDEAL_op(B, te=te, training=False)
    if args.G_model!='MEBCRN':
        A = data.A_from_MEBCRN(A)
        B = data.B_from_MEBCRN(B,mag_and_phase=True)
    if args.sigma_noise > 0.0:
        A = tf.keras.layers.GaussianNoise(stddev=args.sigma_noise)(A, training=True)
    B_WF = B[:,:,:,:4]
    B_PM = B[:,:,:,4:]
    B_WF_abs = tf.abs(tf.complex(B_WF[:,:,:,0::2],B_WF[:,:,:,1::2]))
    with tf.GradientTape() as t:
        if args.out_vars == 'WF':
            # Compute model's output
            A2B_WF_abs = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_WF_abs = tf.where(B[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)

            # Compute zero-valued param maps
            A2B_PM = tf.zeros_like(A2B_WF_abs)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)

        elif args.out_vars == 'WFc':
            # Compute model's output
            A2B_WF = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_WF = tf.where(B[:,:,:,:4]!=0.0,A2B_WF,0.0)

            # Magnitude of water/fat images
            A2B_WF_real = A2B_WF[:,:,:,0::2]
            A2B_WF_imag = A2B_WF[:,:,:,1::2]
            A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))

            # Compute zero-valued param maps
            A2B_PM = tf.zeros_like(A2B_WF_abs)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Compute loss
            sup_loss = sup_loss_fn(B_WF, A2B_WF)

        elif args.out_vars == 'PM':
            # Compute model's output
            A2B_PM = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_PM = tf.where(B[:,:,:,:2]!=0.0,A2B_PM,0.0)

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(B[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)

            # Compute water/fat
            A2B_WF = wf.get_rho(A, A2B_PM, MEBCRN=False)
            
            # Magnitude of water/fat images
            A2B_WF_real = A2B_WF[:,:,:,0::2]
            A2B_WF_imag = A2B_WF[:,:,:,1::2]
            A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
            
            # Compute loss
            sup_loss = sup_loss_fn(B_PM, A2B_PM)

        elif args.out_vars == 'WF-PM':
            # Compute model's output
            B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
            A2B_abs = G_A2B(A, training=True)
            if not(args.DL_gen):
                A2B_abs = tf.where(B[:,:,:,:4]!=0.0,A2B_abs,0.0)

            # Split A2B outputs
            A2B_WF_abs = A2B_abs[:,:,:,:2]
            A2B_PM = A2B_abs[:,:,:,2:]

            # Split A2B param maps
            A2B_R2 = A2B_PM[:,:,:,:1]
            A2B_FM = A2B_PM[:,:,:,1:]

            # Restore field-map when necessary
            if args.G_model=='U-Net' or args.G_model=='MEBCRN':
                A2B_FM = (A2B_FM - 0.5) * 2
                A2B_FM = tf.where(B[:,:,:,:1]!=0.0,A2B_FM,0.0)
                A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)

            # Compute loss
            sup_loss = sup_loss_fn(B_abs, A2B_abs)

        ############### Splited losses ####################
        WF_abs_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
        R2_loss = sup_loss_fn(B_PM[:,:,:,:1], A2B_R2)
        FM_loss = sup_loss_fn(B_PM[:,:,:,1:], A2B_FM)

        ################ Regularizers #####################
        if not(args.out_vars=='WF' or args.out_vars=='WFc'):
            R2_TV = tf.reduce_sum(tf.image.total_variation(A2B_R2)) * args.R2_TV_weight
            FM_TV = tf.reduce_sum(tf.image.total_variation(A2B_FM)) * args.FM_TV_weight
            R2_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_R2),axis=(1,2,3))) * args.R2_L1_weight
            FM_L1 = tf.reduce_sum(tf.reduce_mean(tf.abs(A2B_FM),axis=(1,2,3))) * args.FM_L1_weight
        else:
            R2_TV = 0.0
            FM_TV = 0.0
            R2_L1 = 0.0
            FM_L1 = 0.0
        reg_term = R2_TV + FM_TV + R2_L1 + FM_L1
        
        G_loss = sup_loss + reg_term
        
    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return {'sup_loss': sup_loss,
            'WF_loss': WF_abs_loss,
            'R2_loss': R2_loss,
            'FM_loss': FM_loss,
            'TV_R2': R2_TV,
            'TV_FM': FM_TV,
            'L1_R2': R2_L1,
            'L1_FM': FM_L1}


def train_step(A, B, te=None):
    G_loss_dict = train_G(A, B, te)
    return G_loss_dict


@tf.function
def sample(A, B):
    B_WF = B[:,:,:,:4]
    B_PM = B[:,:,:,4:]
    B_WF_abs = tf.abs(tf.complex(B_WF[:,:,:,0::2],B_WF[:,:,:,1::2]))
    # Estimate A2B
    if args.out_vars == 'WF':
        A2B_WF_abs = G_A2B(A, training=True)
        A2B_WF_abs = tf.where(B[:,:,:,:2]!=0.0,A2B_WF_abs,0.0)
        A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
    elif args.out_vars == 'WFc':
        A2B_WF = G_A2B(A, training=True)
        A2B_WF = tf.where(B[:,:,:,:4]!=0.0,A2B_WF,0.0)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_PM = tf.zeros_like(B_PM)
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_WF, A2B_WF)
    elif args.out_vars == 'PM':
        A2B_PM = G_A2B(A, training=True)
        A2B_PM = tf.where(B_PM!=0.0,A2B_PM,0.0)
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,1:]!=0.0,A2B_FM,0.0)
            A2B_PM = tf.concat([A2B_R2,A2B_FM],axis=-1)
        A2B_WF = wf.get_rho(A,A2B_PM,MEBCRN=False)
        A2B_WF_real = A2B_WF[:,:,:,0::2]
        A2B_WF_imag = A2B_WF[:,:,:,1::2]
        A2B_WF_abs = tf.abs(tf.complex(A2B_WF_real,A2B_WF_imag))
        A2B_abs = tf.concat([A2B_WF_abs,A2B_PM],axis=-1)
        val_sup_loss = sup_loss_fn(B_PM, A2B_PM)
    elif args.out_vars == 'WF-PM':
        B_abs = tf.concat([B_WF_abs,B_PM],axis=-1)
        A2B_abs = G_A2B(A, training=True)
        A2B_abs = tf.where(B_abs!=0.0,A2B_abs,0.0)
        # Split A2B outputs
        A2B_WF_abs = A2B_abs[:,:,:,:2]
        A2B_PM = A2B_abs[:,:,:,2:]
        # Split A2B param maps
        A2B_R2 = A2B_PM[:,:,:,:1]
        A2B_FM = A2B_PM[:,:,:,1:]
        if args.G_model=='U-Net' or args.G_model=='MEBCRN':
            A2B_FM = (A2B_FM - 0.5) * 2
            A2B_FM = tf.where(B_PM[:,:,:,:1]!=0.0,A2B_FM,0.0)
            A2B_abs = tf.concat([A2B_WF_abs,A2B_R2,A2B_FM],axis=-1)
        val_sup_loss = sup_loss_fn(B_abs, A2B_abs)

    ############### Splited losses ####################
    WF_abs_loss = sup_loss_fn(B_WF_abs, A2B_WF_abs)
    R2_loss = sup_loss_fn(B_PM[:,:,:,:1], A2B_R2)
    FM_loss = sup_loss_fn(B_PM[:,:,:,1:], A2B_FM)

    return A2B_abs,{'sup_loss': val_sup_loss,
                    'WF_loss': WF_abs_loss,
                    'R2_loss': R2_loss,
                    'FM_loss': FM_loss}

def validation_step(A, B):
    A2B_abs, val_sup_dict = sample(A, B)
    return A2B_abs, val_sup_dict


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_optimizer=G_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))
val_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'validation'))

# sample
val_iter = cycle(A_B_dataset_val)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)
n_div = np.ceil(total_steps/len(valX))

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for A, B in A_B_dataset:
        if (args.TE1 == 0.0013) and (args.dTE == 0.0021):
            te = None
        else:
            te = wf.gen_TEvar(args.n_echoes, bs=B.shape[0], TE_ini_d=args.TE1, d_TE_min=args.dTE)
        # ==============================================================================
        
        G_loss_dict = train_step(A, B, te)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 200//args.batch_size):
            A, B = next(val_iter)
            # A = A[0,...]
            # B = B[0,...]

            A = tf.expand_dims(A,axis=0)
            B = tf.expand_dims(B,axis=0)
            A2B, val_sup_dict = validation_step(A, B)

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_sup_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            if args.G_model == 'MEBCRN':
                im_ech1 = np.squeeze(np.abs(tf.complex(A[:,0,:,:,0],A[:,0,:,:,1])))
                im_ech2 = np.squeeze(np.abs(tf.complex(A[:,1,:,:,0],A[:,1,:,:,1])))
                if args.n_echoes >= 3:
                    im_ech3 = np.squeeze(np.abs(tf.complex(A[:,2,:,:,0],A[:,2,:,:,1])))
                if args.n_echoes >= 4:
                    im_ech4 = np.squeeze(np.abs(tf.complex(A[:,3,:,:,0],A[:,3,:,:,1])))
                if args.n_echoes >= 5:
                    im_ech5 = np.squeeze(np.abs(tf.complex(A[:,4,:,:,0],A[:,4,:,:,1])))
                if args.n_echoes >= 6:
                    im_ech6 = np.squeeze(np.abs(tf.complex(A[:,5,:,:,0],A[:,5,:,:,1])))
            else:
                im_ech1 = np.squeeze(np.abs(tf.complex(A[:,:,:,0],A[:,:,:,1])))
                im_ech2 = np.squeeze(np.abs(tf.complex(A[:,:,:,2],A[:,:,:,3])))
                if args.n_echoes >= 3:
                    im_ech3 = np.squeeze(np.abs(tf.complex(A[:,:,:,4],A[:,:,:,5])))
                if args.n_echoes >= 4:
                    im_ech4 = np.squeeze(np.abs(tf.complex(A[:,:,:,6],A[:,:,:,7])))
                if args.n_echoes >= 5:
                    im_ech5 = np.squeeze(np.abs(tf.complex(A[:,:,:,8],A[:,:,:,9])))
                if args.n_echoes >= 6:
                    im_ech6 = np.squeeze(np.abs(tf.complex(A[:,:,:,10],A[:,:,:,11])))

            
            # Acquisitions in the first row
            acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,0].set_title('1st Echo')
            axs[0,0].axis('off')
            acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,1].set_title('2nd Echo')
            axs[0,1].axis('off')
            if args.n_echoes >= 3:
                acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,2].set_title('3rd Echo')
                axs[0,2].axis('off')
            else:
                fig.delaxes(axs[0,2])
            if args.n_echoes >= 4:
                acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,3].set_title('4th Echo')
                axs[0,3].axis('off')
            else:
                fig.delaxes(axs[0,3])
            if args.n_echoes >= 5:
                acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,4].set_title('5th Echo')
                axs[0,4].axis('off')
            else:
                fig.delaxes(axs[0,4])
            if args.n_echoes >= 6:
                acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,5].set_title('6th Echo')
                axs[0,5].axis('off')
            else:
                fig.delaxes(axs[0,5])

            # A2B maps in the second row
            w_aux = np.squeeze(A2B[:,:,:,0])
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(A2B[:,:,:,1])
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(A2B[:,:,:,2])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(A2B[:,:,:,3])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,4])
            axs[1,4].axis('off')
            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            wn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,0],B[:,:,:,1])))
            W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            fn_aux = np.squeeze(np.abs(tf.complex(B[:,:,:,2],B[:,:,:,3])))
            F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            r2n_aux = np.squeeze(B[:,:,:,4])
            r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            fieldn_aux = np.squeeze(B[:,:,:,5])
            field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_unet, ax=axs[2,4])
            axs[2,4].axis('off')
            fig.delaxes(axs[2,0])
            fig.delaxes(axs[2,5])

            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)
