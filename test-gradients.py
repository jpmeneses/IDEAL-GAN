import numpy as np
import matplotlib.pyplot as plt
import tqdm

import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import DLlib as dl
import pylib as py
import wflib as wf
import data

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../datasets/'
dataset_hdf5 = 'JGalgani_GC_384_complex_2D.hdf5'
acqs, out_maps = data.load_hdf5(dataset_dir, dataset_hdf5, 12, end=87, MEBCRN=True)
acqs = acqs[:,:,::8,::8,:]
out_maps = out_maps[:,:,::8,::8,:]

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(acqs)
_,n_out,_,_,_ = np.shape(out_maps)
r2_sc = 200.0
fm_sc = 300.0

print('Dataset size:', len_dataset)
print('Acquisition Dimensions:', hgt,wdt)
print('Output Maps:',n_out)

bs = 4
A_B_dataset = tf.data.Dataset.from_tensor_slices((acqs,out_maps))
A_B_dataset = A_B_dataset.batch(bs).shuffle(len_dataset)

G_A2B = dl.UNet(input_shape=(None,None,None,n_ch),
                bayesian=True,
                ME_layer=True,
                filters=36,
                self_attention=True)
G_A2R2= dl.UNet(input_shape=(None,hgt,wdt,1),
                bayesian=True,
                ME_layer=True,
                filters=12,
                output_activation='sigmoid',
                output_initializer='he_uniform',
                self_attention=False)


cycle_loss_fn = tf.losses.MeanSquaredError()
uncertain_loss = gan.VarMeanSquaredError()

G_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
D_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)

A2B2A_pool = data.ItemPool(10)

@tf.function
def train_G(A):
    # te = wf.gen_TEvar(ne,orig=True)
    with tf.GradientTape() as t:
        ##################### A Cycle #####################
        A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A),axis=-1,keepdims=True))

        # Compute R2s map from only-mag images
        A2B_R2, A2B_R2_nu, A2B_R2_sigma = G_A2R2(A_abs, training=True) # Randomly sampled R2s
        A2B_R2 = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2,0.0)

        # Compute FM using complex-valued images and pre-trained model
        A2B_FM, _, A2B_FM_var = G_A2B(A, training=False) # Mean FM
        A2B_FM = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM,0.0)
        A2B_PM = tf.concat([A2B_FM,A2B_R2], axis=-1)

        # Magnitude of water/fat images
        A2B_WF, A2B2A = wf.acq_to_acq(A, A2B_PM)
        A2B = tf.concat([A2B_WF,A2B_PM], axis=1)
        A2B_WF_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B_WF),axis=-1,keepdims=True))
        A2B2A_abs = tf.math.sqrt(tf.reduce_sum(tf.square(A2B2A),axis=-1,keepdims=True))
        
        # Variance map mask and attach to recon-A
        A2B_FM_var = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_FM_var,0.0)
        A2B_R2_nu = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_nu,0.0)
        A2B_R2_sigma = tf.where(A[:,:1,:,:,:1]!=0.0,A2B_R2_sigma,0.0)
        A2B_PM_var = tf.concat([A2B_FM_var,A2B_R2_nu,A2B_R2_sigma], axis=-1)
        A2B2A_var = wf.acq_uncertainty(tf.stop_gradient(A2B), A2B_PM_var, ne=A.shape[1], only_mag=True)
        A2B2A_sampled_var = tf.concat([A2B2A_abs, A2B2A_var], axis=-1) # shape: [nb,ne,hgt,wdt,2]

        ############ Cycle-Consistency Losses #############
        # CHECK
        A2B2A_cycle_loss = uncertain_loss(A_abs, A2B2A_sampled_var)
        
        G_loss = A2B2A_cycle_loss
        
    G_grad = t.gradient(G_loss, G_A2R2.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2R2.trainable_variables))

    return A2B,A2B2A,{'A2B2A_cycle_loss': A2B2A_cycle_loss,}


@tf.function
def train_D(A, A2B2A):
    # Z = tf.random.normal([A.shape[0],A.shape[2]//(2**(4)),A.shape[2]//(2**(4)),64])
    A = tf.reshape(A,(-1,hgt,wdt,2))
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        A2B2A_d_logits = D_A(A2B2A, training=True)
        
        A_d_loss, A2B2A_d_loss = d_loss_fn(A_d_logits, A2B2A_d_logits)
        tf.debugging.check_numerics(A2B2A_d_loss, message='A2B2A numerical error')

        D_loss = (A_d_loss + A2B2A_d_loss)

    D_grad = t.gradient(D_loss, D_A.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables))

    return {'D_loss': A_d_loss + A2B2A_d_loss,
            'A_d_loss': A_d_loss,
            'A2B2A_d_loss': A2B2A_d_loss,}


def train_step(A):
    A2B, A2B2A, G_loss_dict = train_G(A)

    return A2B, A2B2A, G_loss_dict


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_optimizer=G_optimizer,
                                ep_cnt=ep_cnt),
                           py.join('output', 'Unsup-108', 'checkpoints'),
                           max_to_keep=5)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join('output','test-grad','summaries'))
sample_dir = py.join('output', 'test-grad', 'samples_training')
py.mkdir(sample_dir)

# main loop
n_div = len_dataset
A_prev = None
for ep in range(20):
    for A, B in tqdm.tqdm(A_B_dataset, desc='Ep. '+str(ep+1), total=len_dataset//bs):
        # A = tf.expand_dims(A,axis=0)
        # B = tf.expand_dims(B,axis=0)
        
        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        A2B, A2B2A, G_loss_dict = train_step(A)
        A2B2A = tf.reshape(A,[-1,hgt,wdt,n_ch])
        
        # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

        # sample
        if (G_optimizer.iterations.numpy()+10) % n_div == 0:
            fig, axs = plt.subplots(figsize=(20, 9), nrows=3, ncols=6)

            # Magnitude of MR images at each echo
            im_ech1 = np.squeeze(np.abs(tf.complex(A2B2A[0,:,:,0],A2B2A[0,:,:,1])))
            im_ech2 = np.squeeze(np.abs(tf.complex(A2B2A[1,:,:,0],A2B2A[1,:,:,1])))
            im_ech3 = np.squeeze(np.abs(tf.complex(A2B2A[2,:,:,0],A2B2A[2,:,:,1])))
            im_ech4 = np.squeeze(np.abs(tf.complex(A2B2A[3,:,:,0],A2B2A[3,:,:,1])))
            im_ech5 = np.squeeze(np.abs(tf.complex(A2B2A[4,:,:,0],A2B2A[4,:,:,1])))
            im_ech6 = np.squeeze(np.abs(tf.complex(A2B2A[5,:,:,0],A2B2A[5,:,:,1])))
            
            # Acquisitions in the first row
            acq_ech1 = axs[0,0].imshow(im_ech1, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,0].set_title('1st Echo')
            axs[0,0].axis('off')
            acq_ech2 = axs[0,1].imshow(im_ech2, cmap='gist_earth',
                                  interpolation='none', vmin=0, vmax=1)
            axs[0,1].set_title('2nd Echo')
            axs[0,1].axis('off')
            acq_ech3 = axs[0,2].imshow(im_ech3, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,2].set_title('3rd Echo')
            axs[0,2].axis('off')
            acq_ech4 = axs[0,3].imshow(im_ech4, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,3].set_title('4th Echo')
            axs[0,3].axis('off')
            acq_ech5 = axs[0,4].imshow(im_ech5, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,4].set_title('5th Echo')
            axs[0,4].axis('off')
            acq_ech6 = axs[0,5].imshow(im_ech6, cmap='gist_earth',
                                      interpolation='none', vmin=0, vmax=1)
            axs[0,5].set_title('6th Echo')
            axs[0,5].axis('off')
            
            # A2B maps in the second row
            w_aux = np.squeeze(np.abs(tf.complex(A2B[0,0,:,:,0],A2B[0,0,:,:,1])))
            W_ok =  axs[1,1].imshow(w_aux, cmap='bone',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_ok, ax=axs[1,1])
            axs[1,1].axis('off')

            f_aux = np.squeeze(np.abs(tf.complex(A2B[0,1,:,:,0],A2B[0,1,:,:,1])))
            F_ok =  axs[1,2].imshow(f_aux, cmap='pink',
                                    interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_ok, ax=axs[1,2])
            axs[1,2].axis('off')

            r2_aux = np.squeeze(A2B[0,2,:,:,1])
            r2_ok = axs[1,3].imshow(r2_aux*r2_sc, cmap='copper',
                                    interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_ok, ax=axs[1,3])
            axs[1,3].axis('off')

            field_aux = np.squeeze(A2B[0,2,:,:,0])
            field_ok =  axs[1,4].imshow(field_aux*fm_sc, cmap='twilight',
                                        interpolation='none', vmin=-fm_sc/2, vmax=fm_sc/2)
            fig.colorbar(field_ok, ax=axs[1,4])
            axs[1,4].axis('off')
            fig.delaxes(axs[1,0])
            fig.delaxes(axs[1,5])

            # Ground-truth in the third row
            wn_aux = np.squeeze(np.abs(tf.complex(B[0,0,:,:,0],B[0,0,:,:,1])))
            W_unet = axs[2,1].imshow(wn_aux, cmap='bone',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(W_unet, ax=axs[2,1])
            axs[2,1].axis('off')

            fn_aux = np.squeeze(np.abs(tf.complex(B[0,1,:,:,0],B[0,1,:,:,1])))
            F_unet = axs[2,2].imshow(fn_aux, cmap='pink',
                                interpolation='none', vmin=0, vmax=1)
            fig.colorbar(F_unet, ax=axs[2,2])
            axs[2,2].axis('off')

            r2n_aux = np.squeeze(B[0,2,:,:,1])
            r2_unet = axs[2,3].imshow(r2n_aux*r2_sc, cmap='copper',
                                 interpolation='none', vmin=0, vmax=r2_sc)
            fig.colorbar(r2_unet, ax=axs[2,3])
            axs[2,3].axis('off')

            fieldn_aux = np.squeeze(B[0,2,:,:,0])
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

