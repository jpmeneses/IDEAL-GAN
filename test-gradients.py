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
dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
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
fm_sc = 300.0

print('Dataset size:', len_dataset)
print('Acquisition Dimensions:', hgt,wdt)
print('Output Maps:',n_out)

A_B_dataset = tf.data.Dataset.from_tensor_slices((acqs,out_maps))
A_B_dataset = A_B_dataset.batch(4).shuffle(len_dataset)

enc= dl.encoder(input_shape=(6,hgt,wdt,n_ch),
                encoded_dims=64,
                filters=12,
                ls_reg_weight=1e-5,
                NL_self_attention=False)
dec_w  = dl.decoder(encoded_dims=64,
                    output_2D_shape=(hgt,wdt),
                    filters=12,
                    NL_self_attention=False)
dec_f  = dl.decoder(encoded_dims=64,
                    output_2D_shape=(hgt,wdt),
                    filters=12,
                    NL_self_attention=False)
dec_xi = dl.decoder(encoded_dims=64,
                    output_2D_shape=(hgt,wdt),
                    filters=12,
                    NL_self_attention=False)


D_A = dl.PatchGAN(input_shape=(hgt,wdt,2), dim=12, self_attention=False)
# D_Z = dl.CriticZ((hgt//(2**(4)),wdt//(2**(4)),64)) # //(2**(4))

# vgg = keras.applications.vgg19.VGG19()
# metric_vgg = keras.Model(inputs=vgg.inputs, outputs=vgg.layers[3].output)

# metric_model = keras.Sequential()
# metric_model.add(keras.layers.Lambda(lambda x: tf.concat([x,tf.zeros_like(x[:,:,:,:1])],axis=-1)))
# metric_model.add(keras.layers.ZeroPadding2D(padding=(88,88)))
# metric_model.add(metric_vgg)
# b = metric_model(tf.random.normal((1,hgt,wdt,3),dtype=tf.float32))

IDEAL_op = wf.IDEAL_Layer()
LWF_op = wf.LWF_Layer(ne,MEBCRN=True)
F_op = dl.FourierLayer()

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')
cycle_loss_fn = tf.losses.MeanSquaredError()
cosine_loss = tf.losses.CosineSimilarity()

G_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
D_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)

A2B2A_pool = data.ItemPool(10)

@tf.function
def train_G(A, B):
    te = wf.gen_TEvar(ne,orig=True)
    with tf.GradientTape(persistent=True) as t:
        ##################### A Cycle #####################
        A2Z = enc(A, training=True)
        A2Z2B_w = dec_w(A2Z, training=True)
        A2Z2B_f = dec_f(A2Z, training=True)
        A2Z2B_xi= dec_xi(A2Z, training=True)
        A2B = tf.concat([A2Z2B_w,A2Z2B_f,A2Z2B_xi],axis=1)
        A2B2A = IDEAL_op(A2B, te, training=False)

        # A2B_L = tf.concat([A2Z2B_w,A2Z2B_f],axis=1)
        # A2B2A_L = LWF_op(A2B_L)

        ##################### B Cycle #####################
        # B2A = IDEAL_op(B, training=False)
        # B2A2B = G_A2B(B2A, training=True)

        ############## Discriminative Losses ##############
        # A2B2A_d_logits = D_A(A2B2A_L, training=False)
        # A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)
        A2B2A_g_loss = tf.constant(0.0,dtype=tf.float32)

        ############# Fourier Regularization ##############
        A_f = F_op(A, training=False)
        A2B2A_f = F_op(A2B2A, training=False)

        ############ Cycle-Consistency Losses #############
        # A2Y = tf.reshape(Aa,[A.shape[0]*ne,hgt,wdt,n_ch])
        # A2Y_m = metric_model(A2Y, training=False)
        
        # A2B2A2Y = tf.reshape(A2B2A,[A2B2A.shape[0]*ne,hgt,wdt,n_ch])
        # A2B2A2Y_m = metric_model(A2B2A2Y, training=False)
        
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, A2B)
        A2B2A_f_cycle_loss = cycle_loss_fn(A_f, A2B2A_f)

        ################ Regularizers #####################
        activ_reg = tf.add_n(enc.losses)
        
        G_loss = A2B2A_g_loss + (A2B2A_cycle_loss + B2A2B_cycle_loss)*1e1 + activ_reg + A2B2A_f_cycle_loss*1e-3
        
    G_grad = t.gradient(G_loss, enc.trainable_variables + dec_w.trainable_variables + dec_f.trainable_variables + dec_xi.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, enc.trainable_variables + dec_w.trainable_variables + dec_f.trainable_variables + dec_xi.trainable_variables))

    return A2B,A2B2A,{'A2B2A_g_loss': A2B2A_g_loss,
                        'A2B2A_cycle_loss': A2B2A_cycle_loss,
                        'B2A2B_cycle_loss': B2A2B_cycle_loss,
                        'A2B2A_f_cycle_loss': A2B2A_f_cycle_loss,
                        'LS_reg': activ_reg}


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


def train_step(A, B):
    A2B, A2B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B2A = A2B2A_pool(A2B2A)
    if False: #ep >= 0:
        for _ in range(0):
            D_loss_dict = train_D(A, A2B2A)
    else:
        D_aux_val = tf.constant(0.0,dtype=tf.float32)
        D_loss_dict = {'D_loss': D_aux_val, 'A_d_loss': D_aux_val, 'A2B2A_d_loss': D_aux_val}

    return A2B, A2B2A, G_loss_dict, D_loss_dict


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join('output','test-grad','summaries'))
sample_dir = py.join('output', 'test-grad', 'samples_training')
py.mkdir(sample_dir)

# main loop
n_div = len_dataset
A_prev = None
for ep in range(20):
    for A, B in tqdm.tqdm(A_B_dataset, desc='Ep. '+str(ep+1), total=len_dataset):
        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        A2B, A2B2A, G_loss_dict, D_loss_dict = train_step(A, B)
        A2B2A = tf.reshape(A,[-1,hgt,wdt,n_ch])
        
        # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')

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

            r2_aux = np.squeeze(A2B[0,2,:,:,1])*2*np.pi
            r2_ok = axs[1,3].imshow(r2_aux*fm_sc, cmap='twilight',
                                    interpolation='none', vmin=-fm_sc, vmax=fm_sc)
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

            r2n_aux = np.squeeze(B[0,2,:,:,1])*2*np.pi
            r2_unet = axs[2,3].imshow(r2n_aux*fm_sc, cmap='twilight',
                                 interpolation='none', vmin=-fm_sc, vmax=fm_sc)
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

