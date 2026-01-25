import functools

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tf2lib as tl
import tf2gan as gan
import DLlib as dl
import pylib as py
import wflib as wf
import data

from itertools import cycle

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='WF-IDEAL')
py.arg('--train_data', default='HDF5', choices=['HDF5','DICOM','NIFTI'])
py.arg('--dataset_dir', default='../datasets/')
py.arg('--training_mode', default='supervised', choices=['supervised','unsupervised'])
py.arg('--data_aug', type=bool, default=False)
py.arg('--gen_data_aug', type=bool, default=False)
py.arg('--gen_partial_real', type=int, default=0, choices=[0,2,6,10])
py.arg('--gen_filename', default='LDM_ds')
py.arg('--shuffle', type=bool, default=True)
py.arg('--n_echoes', type=int, default=6)
py.arg('--min_rand_ne', type=int, default=4)
py.arg('--max_rand_ne', type=int, default=6)
py.arg('--field', type=float, default=1.5)
py.arg('--n_G_filters', type=int, default=36)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--epoch_ckpt', type=int, default=20)  # num. of epochs to save a checkpoint
py.arg('--lr', type=float, default=0.0008)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--beta_2', type=float, default=0.999)
py.arg('--main_loss', default='Rice', choices=['Rice', 'MSE', 'MAE', 'MSLE'])
py.arg('--R2_TV_weight', type=float, default=0.0)
py.arg('--A_demod_TV_weight', type=float, default=0.0)
py.arg('--D1_SelfAttention',type=bool, default=False)
args = py.args()

# output_dir
output_dir = py.join('output',args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

if args.n_echoes > 0:
    ech_idx = args.n_echoes * 2
else:
    ech_idx = 12
fm_sc = 300.0
if args.field == 3.0:
    r2_sc = 600.0
else:
    r2_sc = 200.0

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################

if args.train_data == 'HDF5':
    dataset_hdf5_1 = 'INTA_GC_384_complex_2D.hdf5'
    valX, valY = data.load_hdf5(args.dataset_dir, dataset_hdf5_1, ech_idx,
                                acqs_data=True, te_data=False, MEBCRN=True)

    len_val = len(valY)
    A_B_dataset_val = tf.data.Dataset.from_tensor_slices((valX, valY))
    A_B_dataset_val.batch(1)

    if not(args.gen_data_aug):
        dataset_hdf5_2 = 'INTArest_GC_384_complex_2D.hdf5'
        dataset_hdf5_3 = 'Volunteers_GC_384_complex_2D.hdf5'
        dataset_hdf5_4 = 'Attilio_GC_384_complex_2D.hdf5'

        if args.training_mode == 'supervised':
            out_maps_2 = data.load_hdf5(args.dataset_dir,dataset_hdf5_2, ech_idx,
                                        acqs_data=False, te_data=False, MEBCRN=True)

            out_maps_3 = data.load_hdf5(args.dataset_dir, dataset_hdf5_3, ech_idx,
                                        acqs_data=False, te_data=False, MEBCRN=True)

            out_maps_4 = data.load_hdf5(args.dataset_dir, dataset_hdf5_4, ech_idx,
                                        acqs_data=False, te_data=False, MEBCRN=True)
        else:
            acqs_2, out_maps_2 = data.load_hdf5(args.dataset_dir,dataset_hdf5_2, ech_idx,
                                                acqs_data=True, te_data=False, MEBCRN=True)

            acqs_3, out_maps_3 = data.load_hdf5(args.dataset_dir, dataset_hdf5_3, ech_idx,
                                                acqs_data=True, te_data=False, MEBCRN=True)

            acqs_4, out_maps_4 = data.load_hdf5(args.dataset_dir, dataset_hdf5_4, ech_idx,
                                                acqs_data=True, te_data=False, MEBCRN=True)
            trainX = np.concatenate((acqs_2,acqs_3,acqs_4),axis=0)

        trainY  = np.concatenate((out_maps_2,out_maps_3,out_maps_4),axis=0)
        len_dataset,n_out,hgt,wdt,n_ch = np.shape(trainY)
        
        if args.training_mode == 'supervised':
            A_B_dataset = tf.data.Dataset.from_tensor_slices(trainY)
        else:
            A_B_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

    else:
        c_pha = 3
        recordPath = py.join('tfrecord', args.DL_filename)
        tfr_dataset = tf.data.TFRecordDataset([recordPath])
        # Create a description of the features.
        feature_description = {
            'out_maps': tf.io.FixedLenFeature([], tf.string),
            }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            parsed_ds = tf.io.parse_example(example_proto, feature_description)
            return tf.io.parse_tensor(parsed_ds['out_maps'], out_type=tf.float32)

        if args.DL_partial_real != 0:
            if args.DL_partial_real == 2:
                end_idx = 62
            elif args.DL_partial_real == 6:
                end_idx = 200
            elif args.DL_partial_real == 10:
                end_idx = 330
            dataset_hdf5_2 = 'INTArest_GC_384_complex_2D.hdf5'
            trainY = data.load_hdf5(args.dataset_dir,dataset_hdf5_2, ech_idx, end=end_idx,
                                    acqs_data=False, te_data=False, MEBCRN=True,
                                    mag_and_phase=True, unwrap=True)
            A_B_dataset = tfr_dataset.skip(end_idx).map(_parse_function)
            A_B_dataset_aux = tf.data.Dataset.from_tensor_slices(trainY)
            A_B_dataset = A_B_dataset.concatenate(A_B_dataset_aux)
        else:
            A_B_dataset = tfr_dataset.map(_parse_function)

        for B in A_B_dataset.take(1):
            n_ch,hgt,wdt,n_out = B.shape
            print(B.shape)
        len_dataset = int(args.DL_filename.split('_')[-1])
        if args.DL_partial_real != 0:
            len_dataset += trainY.shape[0]

    A_B_dataset = A_B_dataset.batch(args.batch_size)
    if args.shuffle:
        A_B_dataset = A_B_dataset.shuffle(len_dataset)

else:
    folders = [os.path.join(args.dataset_dir, d) for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))]
    folders_mr = [os.path.join(f, os.listdir(f)[0]) for i, f in enumerate(folders) if os.path.join(f, os.listdir(f)[0])]
    folders_cse = list()
    for f in folders_mr:
        scan_files = os.listdir(f)
        if args.train_data == 'DICOM':
            cse_scan = [item for item in scan_files if "MECSE" in item]
        elif args.train_data == 'NIFTI':
            cse_scan = [item for item in scan_files if "nifti" in item]
        folders_cse.append(os.path.join(f,cse_scan[0]))
    num_fold = len(folders_cse)

    A_B_dataset = tf.data.Dataset.from_tensor_slices(folders_cse[(num_fold//6):])
    if args.train_data == 'DICOM':
        A_B_dataset = A_B_dataset.map(lambda f: data.tf_load_dicom_series(f))
    elif args.train_data == 'NIFTI':
        A_B_dataset = A_B_dataset.map(lambda f: data.tf_load_nifti_series(f))
    A_B_dataset = A_B_dataset.unbatch()

    len_dataset = sum(1 for _ in A_B_dataset)
    for a in A_B_dataset.take(1):
        ne,hgt,wdt,n_ch = a.shape
    A_B_dataset = A_B_dataset.batch(args.batch_size).shuffle(len_dataset)

    A_B_dataset_val = tf.data.Dataset.from_tensor_slices(folders_cse[:(num_fold//6)])
    if args.train_data == 'DICOM':
        A_B_dataset_val = A_B_dataset_val.map(lambda f: data.tf_load_dicom_series(f))
    elif args.train_data == 'NIFTI':
        A_B_dataset_val = A_B_dataset_val.map(lambda f: data.tf_load_nifti_series(f))
    A_B_dataset_val = A_B_dataset_val.unbatch()
    len_val = sum(1 for _ in A_B_dataset_val)
    A_B_dataset_val = A_B_dataset_val.batch(1)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

total_steps = np.ceil(len_dataset/args.batch_size)*args.epochs

G_mag = dl.UNet(input_shape=(None,hgt,wdt,1),
                bayesian=(args.main_loss=='Rice'),
                ME_layer=True,
                te_input=(args.n_echoes==0),
                te_shape=(None,),
                filters=args.n_G_filters,
                output_activation='sigmoid',
                self_attention=args.D1_SelfAttention)

IDEAL_op = wf.IDEAL_Layer(field=args.field, r2_sc=r2_sc)

if args.main_loss == 'Rice':
    loss_fn = lambda y, p_y: -p_y.log_prob(y)
    loss_alt = tf.losses.MeanSquaredError()
elif args.main_loss == 'MSE':
    loss_fn = loss_alt = tf.losses.MeanSquaredError()
elif args.main_loss == 'MAE':
    loss_fn = loss_alt = tf.losses.MeanAbsoluteError()
elif args.main_loss == 'MSLE':
    loss_fn = loss_alt = tf.losses.MeanSquaredLogarithmicError()
else:
    raise(NameError('Unrecognized Main Loss Function'))

G_lr_scheduler = dl.LinearDecay(args.lr, args.epochs, args.epoch_decay)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1, beta_2=args.beta_2)

# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(B, A=None, te=None):
    if A is None:
        A = IDEAL_op(B, te=te, training=False)
    A_mag = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
    with tf.GradientTape() as t:
        # Compute model's output
        if args.n_echoes==0:
            A2B_R2 = G_mag([A_mag, te], training=True)
        else:
            A2B_R2 = G_mag(A_mag, training=True)
        if args.main_loss != 'Rice':
            A2B_R2 = tf.where(A_mag[:,:1,...]!=0.0,A2B_R2,0.0)

        A2B_WF_mag, A2B2A_mag, A_demod = wf.CSE_mag(A_mag, A2B_R2, [args.field, te],
                                                    demod_signal=True, R2_prob=(args.main_loss=='Rice'))
        A2B2A_mag = tf.where(A_mag!=0.0,A2B2A_mag,0.0)

        A2B2A_cycle_loss = loss_alt(A_mag, A2B2A_mag)

        ############### Splited losses ####################
        if B is not None:
            B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,...]),axis=-1,keepdims=True))
            A2B_WF_mag = tf.where(B_WF_abs!=0.0,A2B_WF_mag,0.0)
            WF_abs_loss = loss_alt(B_WF_abs, A2B_WF_mag)
            R2_loss = loss_fn(B[:,2:,:,:,1:], A2B_R2)

            if args.main_loss == 'Rice':
                R2_TV_aux = A2B_R2.nu
            else:
                R2_TV_aux = A2B_R2
            R2_TV = tf.reduce_sum(tf.image.total_variation(R2_TV_aux[:,0,...]))

        else:
            WF_abs_loss = tf.constant(0.0)
            R2_loss = tf.constant(0.0)
            R2_TV = tf.constant(0.0)

        if args.training_mode == 'supervised':
            G_loss = R2_loss
        elif args.training_mode == 'unsupervised':
            G_loss = A2B2A_cycle_loss

        G_loss += R2_TV * args.R2_TV_weight
        
        Ad_aux = tf.reshape(A_demod,[-1,A2B2A_mag.shape[2],A2B2A_mag.shape[3],A2B2A_mag.shape[4]])
        Ad_TV = tf.reduce_sum(tf.image.total_variation(Ad_aux))
        G_loss += Ad_TV * args.A_demod_TV_weight
        
    G_grad = t.gradient(G_loss, G_mag.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_mag.trainable_variables))

    return {'A2B2A_cycle_loss': A2B2A_cycle_loss,
            'WF_loss': WF_abs_loss,
            'R2_loss': R2_loss,
            'R2_TV': R2_TV,
            'Ad_TV': Ad_TV}


def train_step(B, A=None, te=None):
    G_loss_dict = train_G(B, A, te)
    return G_loss_dict


@tf.function
def sample(B, A=None, te=None):
    if A is None:
        A = IDEAL_op(B, te=te, training=False)
    A_mag = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))
    
    # Compute model's output
    if args.n_echoes==0:
        A2B_R2 = G_mag([A_mag, te], training=False)
    else:
        A2B_R2 = G_mag(A_mag, training=False)
    if args.main_loss != 'Rice':
        A2B_R2 = tf.where(A_mag[:,:1,...]!=0.0,A2B_R2,0.0)

    A2B_WF_mag, A2B2A_mag = wf.CSE_mag(A_mag, A2B_R2, [args.field, te], r2_sc=r2_sc)
    A2B2A_mag = tf.where(A_mag!=0.0,A2B2A_mag,0.0)
    A2B = tf.concat([A2B_WF_mag,A2B_R2], axis=1)

    A2B2A_cycle_loss = loss_alt(A_mag, A2B2A_mag)

    ############### Splited losses ####################
    if B is not None:
        B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B[:,:2,...]),axis=-1,keepdims=True))
        B_abs = tf.concat([B_WF_abs,B[:,2:,:,:,1:]],axis=1)
        A2B_WF_mag = tf.where(B_WF_abs!=0.0,A2B_WF_mag,0.0)
        WF_abs_loss = loss_alt(B_WF_abs, A2B_WF_mag[:,:1,:,:,:2])
        R2_loss = loss_alt(B[:,2:,:,:,1:], A2B_R2)
        if args.main_loss == 'Rice':
            A2B = tf.where(B_abs!=0.0,A2B,0.0)
    else:
        WF_abs_loss = tf.constant(0.0)
        R2_loss = tf.constant(0.0)

    return A2B2A_mag, A2B, {'A2B2A_cycle_loss': A2B2A_cycle_loss,
                            'WF_loss': WF_abs_loss,
                            'R2_loss': R2_loss}


def validation_step(B, A=None, te=None):
    A2B2A_mag, A2B, val_B2A2B_dict = sample(B, A, te)
    return A2B2A_mag, A2B, val_B2A2B_dict

# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_mag=G_mag,
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
n_div = np.ceil(total_steps/len_val)

# main loop
for ep in range(args.epochs):
    if ep < ep_cnt:
        continue

    # update epoch counter
    ep_cnt.assign_add(1)

    # train for an epoch
    for X in A_B_dataset:
        if len(X) > 1:
            A = X[0]
            B = X[1]
            bs = A.shape[0]
        elif X.shape[1] >= 6:
            A = X
            B = None
            bs = X.shape[0]
        else:
            A = None
            B = X
            bs = X.shape[0]
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        if args.data_aug:
            p = np.random.rand()
            if p <= 0.4:
                B = tf.reshape(tf.transpose(B,perm=[0,2,3,1,4]),[bs,hgt,wdt,n_out*n_ch])
                
                # Random 90 deg rotations
                B = tf.image.rot90(B,k=np.random.randint(3))

                # Random horizontal reflections
                B = tf.image.random_flip_left_right(B)

                # Random vertical reflections
                B = tf.image.random_flip_up_down(B)

                if args.gen_data_aug:
                    B = tf.transpose(tf.reshape(B,[bs,hgt,wdt,n_ch,n_out]),[0,3,1,2,4])
                else:
                    B = tf.transpose(tf.reshape(B,[bs,hgt,wdt,n_out,n_ch]),[0,3,1,2,4])
        
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        if args.n_echoes == 0:
            ne_sel = np.random.randint(args.min_rand_ne,args.max_rand_ne+1)
        else:
            ne_sel = 0
        if args.field == 3.0:
            if args.n_echoes==0:
                te_var=wf.gen_TEvar(args.n_echoes+ne_sel, bs=bs, TE_ini_min=0.8e-3,
                                    TE_ini_d=0.4e-3, d_TE_min=0.6e-3, d_TE_d=0.4e-3)
            else:
                te_var=wf.gen_TEvar(args.n_echoes+ne_sel, bs=bs, TE_ini_min=0.879e-3,
                                    TE_ini_d=None, d_TE_min=0.662e-3, d_TE_d=None)
        else:
            te_var = wf.gen_TEvar(args.n_echoes+ne_sel, bs=bs)

        G_loss_dict = train_step(B, A, te=te_var)

        # # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary({'G learning rate': G_lr_scheduler.current_learning_rate}, 
                        step=G_optimizer.iterations, name='G learning rate')

        # sample
        if (G_optimizer.iterations.numpy() % n_div == 0) or (G_optimizer.iterations.numpy() < 100//args.batch_size):
            X = next(val_iter)
            if len(X) > 1:
                A = tf.expand_dims(X[0], axis=0)
                B = tf.expand_dims(X[1], axis=0)
                ne_sel_val = np.random.randint(args.min_rand_ne,A.shape[0]+1)
                A = A[:,:ne_sel_val,...]
                B_WF = B[:,:2,:,:,:]
                B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))
                TE_valid = wf.gen_TEvar(ne_sel_val, 1, orig=True)
            elif X.shape[1] >= 6:
                A = X
                B = None
                ne_sel_val = np.random.randint(args.min_rand_ne,A.shape[0]+1)
                if args.field == 3.0:
                    TE_valid=wf.gen_TEvar(ne_sel_val, bs=A.shape[0], TE_ini_min=0.879e-3, 
                                        TE_ini_d=None, d_TE_min=0.662e-3, d_TE_d=None)
                else:
                    TE_valid = wf.gen_TEvar(ne_sel_val, bs=A.shape[0], orig=True)
                A = A[:,:ne_sel_val,...]
            else:
                A = None
                B = tf.expand_dims(X, axis=0)
                B_WF = B[:,:2,:,:,:]
                B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(B_WF),axis=-1,keepdims=True))
                ne_sel_val = np.random.randint(args.min_rand_ne,args.max_rand_ne+1)
                if args.field == 3.0:
                    TE_valid = wf.gen_TEvar(ne_sel_val, 1, TE_ini_d=0.4e-3, d_TE_min=1.0e-3, d_TE_d=0.3e-3)
                else:
                    TE_valid = wf.gen_TEvar(ne_sel_val, 1, orig=True)
            
            B2A, B2A2B, val_A2B_dict = validation_step(B, A, te=TE_valid)
            B2A2B_WF_abs = B2A2B[:,:2,:,:,:]

            # # summary
            with val_summary_writer.as_default():
                tl.summary(val_A2B_dict, step=G_optimizer.iterations, name='G_losses')

            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of recon MR images at each echo
            im_ech1 = np.squeeze(B2A[:,0,...])
            im_ech2 = np.squeeze(B2A[:,1,...])
            if B2A.shape[1] >= 3:
                im_ech3 = np.squeeze(B2A[:,2,...])
            if B2A.shape[1] >= 4:
                im_ech4 = np.squeeze(B2A[:,3,...])
            if B2A.shape[1] >= 5:
                im_ech5 = np.squeeze(B2A[:,4,...])
            if B2A.shape[1] >= 6:
                im_ech6 = np.squeeze(B2A[:,5,...])
            
            # Acquisitions in the first row
            acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,0].set_title('1st Echo')
            axs[0,0].axis('off')
            acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,1].set_title('2nd Echo')
            axs[0,1].axis('off')
            if B2A.shape[1] >= 3:
                acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,2].set_title('3rd Echo')
                axs[0,2].axis('off')
            else:
                fig.delaxes(axs[0,2])
            if B2A.shape[1] >= 4:
                acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,3].set_title('4th Echo')
                axs[0,3].axis('off')
            else:
                fig.delaxes(axs[0,3])
            if B2A.shape[1] >= 5:
                acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,4].set_title('5th Echo')
                axs[0,4].axis('off')
            else:
                fig.delaxes(axs[0,4])
            if B2A.shape[1] >= 6:
                acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
                axs[0,5].set_title('6th Echo')
                axs[0,5].axis('off')
            else:
                fig.delaxes(axs[0,5])

            # B2A2B maps in the second row
            w_aux = np.squeeze(B2A2B_WF_abs[:,0,...])
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(B2A2B_WF_abs[:,1,...])
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(B2A2B[:,2,...])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,4])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            if B is not None:
                wn_aux = np.squeeze(B_WF_abs[:,0,:,:,:])
                W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
                fig.colorbar(W_unet, ax=axs[2,1])
                axs[2,1].axis('off')

                fn_aux = np.squeeze(B_WF_abs[:,1,:,:,:])
                F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
                fig.colorbar(F_unet, ax=axs[2,2])
                axs[2,2].axis('off')

                r2n_aux = np.squeeze(B[:,2,:,:,1])
                r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                     interpolation='none', vmin=0, vmax=r2_sc)
                fig.colorbar(r2_unet, ax=axs[2,3])
                axs[2,3].axis('off')

                fieldn_aux = np.squeeze(B[:,2,:,:,0])
                field_unet = axs[2,4].imshow(fieldn_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
                fig.colorbar(field_unet, ax=axs[2,4])
                axs[2,4].axis('off')
            else:
                fig.delaxes(axs[2,1])
                fig.delaxes(axs[2,2])
                fig.delaxes(axs[2,3])
                fig.delaxes(axs[2,4])
            fig.delaxes(axs[2,0])
            fig.delaxes(axs[2,5])

            fig.suptitle('TE1/dTE: '+str([TE_valid[0,0,0].numpy(),np.mean(np.diff(TE_valid,axis=1))]), fontsize=16)

            # plt.show()
            plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
            tl.make_space_above(axs,topmargin=0.8)
            plt.savefig(py.join(sample_dir, 'iter-%09d.png' % G_optimizer.iterations.numpy()),
                        bbox_inches = 'tight', pad_inches = 0)
            plt.close(fig)

    # save checkpoint
    if (((ep+1) % args.epoch_ckpt) == 0) or ((ep+1)==args.epochs):
        checkpoint.save(ep)