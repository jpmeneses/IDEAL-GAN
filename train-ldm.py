import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import openpyxl

import tf2lib as tl
import DLlib as dl
import DMlib as dm
import pylib as py
import wflib as wf
import data

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default='output/GAN-100')
py.arg('--conditional', type=bool, default=False)
py.arg('--scheduler', default='linear', choices=['linear','cosine'])
py.arg('--n_timesteps', type=int, default=200)
py.arg('--beta_start', type=float, default=0.0001)
py.arg('--beta_end', type=float, default=0.02)
py.arg('--s_value', type=float, default=8e-3)
py.arg('--n_ldm_filters', type=int, default=64)
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs_ldm', type=int, default=100)
py.arg('--epoch_ldm_ckpt', type=int, default=10)  # num. of epochs to save a checkpoint
py.arg('--data_augmentation', type=bool, default=False)
py.arg('--lr', type=float, default=0.00005)
ldm_args = py.args()

output_dir = ldm_args.experiment_dir
args = py.args_from_yaml(py.join(output_dir, 'settings.yml'))
args.__dict__.update(ldm_args.__dict__)

if not(hasattr(args,'VQ_num_embed')):
    py.arg('--VQ_num_embed', type=int, default=256)
    py.arg('--VQ_commit_cost', type=float, default=0.5)
    VQ_args = py.args()
    args.__dict__.update(VQ_args.__dict__)

if not(hasattr(args,'unwrap')):
    py.arg('--unwrap', type=bool, default=False)
    UW_args = py.args()
    args.__dict__.update(UW_args.__dict__)

if hasattr(args,'n_G_filt_list'):
    if len(args.n_G_filt_list) > 0:
        filt_list = [int(a_i) for a_i in args.n_G_filt_list.split(',')]
    else:
        filt_list = list()

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5_1 = 'INTArest_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_1, out_maps_1 = data.load_hdf5(dataset_dir,dataset_hdf5_1, MEBCRN=True,
                                    mag_and_phase=args.only_mag, unwrap=args.unwrap)

dataset_hdf5_2 = 'Volunteers_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_2, out_maps_2 = data.load_hdf5(dataset_dir,dataset_hdf5_2, MEBCRN=True,
                                    mag_and_phase=args.only_mag, unwrap=args.unwrap)

dataset_hdf5_3 = 'Attilio_GC_' + str(args.data_size) + '_complex_2D.hdf5'
acqs_3, out_maps_3 = data.load_hdf5(dataset_dir,dataset_hdf5_3, MEBCRN=True,
                                    mag_and_phase=args.only_mag, unwrap=args.unwrap)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

trainX  = np.concatenate((acqs_1,acqs_2,acqs_3),axis=0)
trainY  = np.concatenate((out_maps_1,out_maps_2,out_maps_3),axis=0)

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(trainX)
if args.only_mag:
    _,_,_,_,n_out = np.shape(trainY)
else:
    _,n_out,_,_,_ = np.shape(trainY)

print('Num. Training slices:', len_dataset)
print('Image Dimensions:', hgt, wdt)
print('Num. Output Maps:',n_out)

xlsx_file_1 = py.join('..','datasets','PDFF-training.xlsx')
wb = openpyxl.load_workbook(xlsx_file_1, data_only=True)
sheet_ranges = wb['Sheet1']

sg_labels = list()
al_labels = list()
for n, i in enumerate(sheet_ranges['E']):
    n_sl = sheet_ranges['F'][n].value
    if n>0:
        sg_labels += [i.value]*int(n_sl)
        al_labels += [0]*(int(n_sl/4)+1)
        al_labels += [1]*(n_sl-2*int(n_sl/4)-1)
        al_labels += [2]*(int(n_sl/4))
sg_labels = np.array(sg_labels)
al_labels = np.array(al_labels)
A_dataset = tf.data.Dataset.from_tensor_slices((trainX,al_labels,sg_labels))
A_dataset = A_dataset.batch(args.batch_size).shuffle(len_dataset)

# ==============================================================================
# =                                   models                                   =
# ==============================================================================

nd = 2
if len(args.n_G_filt_list) == (args.n_downsamplings+1):
    nfe = filt_list
    nfd = [a//nd for a in filt_list]
    nfd2 = [a//(nd+1) for a in filt_list]
else:
    nfe = args.n_G_filters
    nfd = args.n_G_filters//nd
    nfd2= args.n_G_filters//(nd+1)
enc= dl.encoder(input_shape=(None,hgt,wdt,n_ch),
                encoded_dims=args.encoded_size,
                filters=nfe,
                num_layers=args.n_downsamplings,
                num_res_blocks=args.n_res_blocks,
                sd_out=not(args.VQ_encoder),
                ls_mean_activ=None,
                NL_self_attention=args.NL_SelfAttention
                )
dec_mag = dl.decoder(encoded_dims=args.encoded_size,
                    output_shape=(hgt,wdt,n_out),
                    filters=nfd,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation='relu',
                    NL_self_attention=args.NL_SelfAttention
                    )
dec_pha = dl.decoder(encoded_dims=args.encoded_size,
                    output_shape=(hgt,wdt,n_out-1),
                    filters=nfd2,
                    num_layers=args.n_downsamplings,
                    num_res_blocks=args.n_res_blocks,
                    output_activation='tanh',
                    NL_self_attention=args.NL_SelfAttention
                    )

# create our unet model
if args.conditional:
    num_classes = int(np.max(al_labels))
else:
    num_classes = None
unet =  dl.denoise_Unet(dim=args.n_ldm_filters,
                        dim_mults=(1,2,4), 
                        channels=args.encoded_size,
                        num_classes=num_classes)

IDEAL_op = wf.IDEAL_mag_Layer()
vq_op = dl.VectorQuantizer(args.encoded_size, args.VQ_num_embed, args.VQ_commit_cost)

tl.Checkpoint(dict(enc=enc,dec_mag=dec_mag,dec_pha=dec_pha,vq_op=vq_op), py.join(args.experiment_dir, 'checkpoints')).restore()

################################################################################
########################### DIFFUSION TIMESTEPS ################################
################################################################################

# create a fixed beta schedule
if args.scheduler == 'linear':
    beta = np.linspace(args.beta_start, args.beta_end, args.n_timesteps)
    # this will be used as discussed in the reparameterization trick
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha, 0)
    alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
elif args.scheduler == 'cosine':
    x = np.linspace(0, args.n_timesteps, args.n_timesteps + 1)
    alpha_bar = np.cos(((x / args.n_timesteps) + args.s_value) / (1 + args.s_value) * np.pi * 0.5) ** 2
    alpha_bar /= alpha_bar[0]
    alpha = np.clip(alpha_bar[1:] / alpha_bar[:-1], 0.0001, 0.9999)
    beta = 1.0 - alpha

# initialize the model in the memory of our GPU
hgt_ls = dec_mag.input_shape[1]
wdt_ls = dec_mag.input_shape[2]
test_images = tf.ones((args.batch_size, hgt_ls, wdt_ls, args.encoded_size), dtype=tf.float32)
test_timestamps = dm.generate_timestamp(0, 1, args.n_timesteps)
test_label = np.random.randint(3, size=(1,), dtype=np.int32)
k = unet(test_images, test_timestamps, test_label)

loss_fn = tf.losses.MeanSquaredError()

# create our optimizer, we will use adam with a Learning rate of 1e-4
opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

def train_step(A, label, Z_std=1.0):
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = dm.generate_timestamp(tsrng, A.shape[0], args.n_timesteps)

    A2Z = enc(A, training=False)
    # A2Z = A2Z.sample()
    A2Z = tf.math.divide_no_nan(A2Z,Z_std)
    A2Z_std = tf.math.reduce_std(A2Z) # For monitoring only

    Z_n, noise = dm.forward_noise(rng, A2Z, timestep_values, alpha_bar)

    with tf.GradientTape() as t:
        pred_noise = unet(Z_n, timestep_values, label)
        
        loss_value = loss_fn(noise, pred_noise)
    
    gradients = t.gradient(loss_value, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return {'Loss': loss_value, 'A2Z_std': A2Z_std}

def validation_step(Z, label, Z_std=1.0):
    for i in range(args.n_timesteps-1):
        t = np.expand_dims(np.array(args.n_timesteps-i-1, np.int32), 0)
        pred_noise = unet(Z, t, label)
        Z = dm.ddpm(Z, pred_noise, t, alpha, alpha_bar, beta)

    if args.VQ_encoder:
        vq_dict = vq_op(Z)
        Z = vq_dict['quantize']
    Z = tf.math.multiply_no_nan(Z,Z_std)
    Z2B_mag = dec_mag(Z, training=False)
    Z2B_pha = dec_pha(Z, training=False)
    Z2B_pha = tf.concat([Z2B_pha[:,:,:,:,:1],Z2B_pha],axis=-1)
    Z2B = tf.concat([Z2B_mag,Z2B_pha],axis=1)
    Z2B2A = IDEAL_op(Z2B, training=False)

    return Z2B, Z2B2A

# epoch counter
ep_cnt_ldm = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# LS scaling factor
z_std_flag = False
z_std = tf.Variable(initial_value=0.0, trainable=False, dtype=tf.float32)

# checkpoint
checkpoint_ldm = tl.Checkpoint(dict(unet=unet,
                                    optimizer=opt,
                                    ep_cnt=ep_cnt_ldm,
                                    z_std=z_std),
                               py.join(output_dir, 'checkpoints_ldm'),
                               max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint_ldm.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'LDM'))

# sample
sample_dir = py.join(output_dir, 'samples_ldm_training')
py.mkdir(sample_dir)

fm_sc = 300.0
r2_sc = 2*np.pi*fm_sc

# Calculate scaling factor (for unitary std dev)
z_num = 0
z_mean = 0.0
z_var = 0.0
if args.VQ_encoder:
    z_std.assign_add(10.0)
elif z_std.numpy() == 0.0:
    for k in range(2):
        for A in A_dataset:
            A2Z = enc(A, training=False)
            if k == 0:
                z_mean += tf.reduce_sum(A2Z)
                z_num += tf.reduce_prod(tf.cast(A2Z.shape,tf.float32))
            else:
                z_var += tf.reduce_sum(tf.square(A2Z - z_mean))
        if k == 0:
            z_mean /= z_num
        else:
            z_var /= z_num
    z_std.assign(tf.math.sqrt(z_var))
        

# main loop
for ep in range(args.epochs_ldm):
    if ep < ep_cnt_ldm:
        continue

    # update epoch counter
    ep_cnt_ldm.assign_add(1)

    # train for an epoch
    for A, lv, sg in A_dataset:
        # ==============================================================================
        # =                             DATA AUGMENTATION                              =
        # ==============================================================================
        if args.data_augmentation:
            p = np.random.rand()
            if p <= 0.4:
                A = tf.reshape(tf.transpose(A,perm=[0,2,3,1,4]),[A.shape[0],hgt,wdt,ne*n_ch])

                # Random 90 deg rotations
                A = tf.image.rot90(A,k=np.random.randint(3))

                # Random horizontal reflections
                A = tf.image.random_flip_left_right(A)

                # Random vertical reflections
                A = tf.image.random_flip_up_down(A)

                A = tf.transpose(tf.reshape(A,[A.shape[0],hgt,wdt,ne,n_ch]),[0,3,1,2,4])
        # ==============================================================================

        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        if args.VQ_encoder:
            loss_dict = train_step(A, lv)
        else:
            loss_dict = train_step(A, lv, z_std)

        # summary
        with train_summary_writer.as_default():
            tl.summary(loss_dict, step=opt.iterations, name='LDM_losses')

    if (ep == 0) or (((ep+1) % args.epoch_ldm_ckpt) == 0) or ((ep+1)==args.epochs_ldm):
        checkpoint_ldm.save(ep)

    # Validation inference
    if (((ep+1) % 20) == 0) or ((ep+1)==args.epochs_ldm):
        Z = tf.random.normal((1,hgt_ls,wdt_ls,args.encoded_size), dtype=tf.float32)
        Lv= np.random.randint(3, size=(1,), dtype=np.int32)
        if args.VQ_encoder:
            Z2B, Z2B2A = validation_step(Z, Lv)
        else:
            Z2B, Z2B2A = validation_step(Z, Lv, z_std)

        fig, axs = plt.subplots(figsize=(20, 6), nrows=2, ncols=6)

        # Magnitude of recon MR images at each echo
        im_ech1 = np.squeeze(np.abs(tf.complex(Z2B2A[:,0,:,:,0],Z2B2A[:,0,:,:,1])))
        im_ech2 = np.squeeze(np.abs(tf.complex(Z2B2A[:,1,:,:,0],Z2B2A[:,1,:,:,1])))
        im_ech3 = np.squeeze(np.abs(tf.complex(Z2B2A[:,2,:,:,0],Z2B2A[:,2,:,:,1])))
        im_ech4 = np.squeeze(np.abs(tf.complex(Z2B2A[:,3,:,:,0],Z2B2A[:,3,:,:,1])))
        im_ech5 = np.squeeze(np.abs(tf.complex(Z2B2A[:,4,:,:,0],Z2B2A[:,4,:,:,1])))
        im_ech6 = np.squeeze(np.abs(tf.complex(Z2B2A[:,5,:,:,0],Z2B2A[:,5,:,:,1])))

        # Acquisitions in the first row
        acq_ech1 =  axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,0].set_title('1st Echo')
        axs[0,0].axis('off')
        acq_ech2 =  axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,1].set_title('2nd Echo')
        axs[0,1].axis('off')
        acq_ech3 =  axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,2].set_title('3rd Echo')
        axs[0,2].axis('off')
        acq_ech4 =  axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,3].set_title('4th Echo')
        axs[0,3].axis('off')
        acq_ech5 =  axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,4].set_title('5th Echo')
        axs[0,4].axis('off')
        acq_ech6 =  axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                    interpolation='none', vmin=0, vmax=1)
        axs[0,5].set_title('6th Echo')
        axs[0,5].axis('off')

        # A2B maps in the second row
        w_m_aux = np.squeeze(Z2B[:,0,:,:,0])
        w_p_aux = np.squeeze(Z2B[:,1,:,:,0])
        f_m_aux = np.squeeze(Z2B[:,0,:,:,1])
        f_p_aux = np.squeeze(Z2B[:,1,:,:,1])
        r2_aux = np.squeeze(Z2B[:,0,:,:,2])
        field_aux = np.squeeze(Z2B[:,1,:,:,2])
        
        W_ok =  axs[1,0].imshow(w_m_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(W_ok, ax=axs[1,0])
        axs[1,0].axis('off')

        Wp_ok = axs[1,1].imshow(w_p_aux, cmap='twilight',
                                interpolation='none', vmin=-1, vmax=1)
        fig.colorbar(Wp_ok, ax=axs[1,1])
        axs[1,1].axis('off')

        F_ok =  axs[1,2].imshow(f_m_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
        fig.colorbar(F_ok, ax=axs[1,2])
        axs[1,2].axis('off')

        Fp_ok = axs[1,3].imshow(f_p_aux, cmap='twilight',
                                interpolation='none', vmin=-1, vmax=1)
        fig.colorbar(Fp_ok, ax=axs[1,3])
        axs[1,3].axis('off')

        r2_ok = axs[1,4].imshow(r2_aux*r2_sc, cmap='copper',
                                interpolation='none', vmin=0, vmax=r2_sc)
        fig.colorbar(r2_ok, ax=axs[1,4])
        axs[1,4].axis('off')

        field_ok =  axs[1,5].imshow(field_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
        fig.colorbar(field_ok, ax=axs[1,5])
        axs[1,5].axis('off')

        fig.suptitle('Slice Level: '+str(Lv[0]), fontsize=16)

        plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.1,wspace=0)
        tl.make_space_above(axs,topmargin=0.8)
        plt.savefig(py.join(sample_dir, 'ep-%05d.png' % (ep+1)), bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)
    