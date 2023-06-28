import numpy as np
import matplotlib.pyplot as plt
import tqdm

import DLlib as dl
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import wflib as wf
import data

################################################################################
######################### DIRECTORIES AND FILENAMES ############################
################################################################################
dataset_dir = '../../OneDrive - Universidad Católica de Chile/Documents/datasets/'
dataset_hdf5 = 'INTA_GC_192_complex_2D.hdf5'
acqs, out_maps = data.load_hdf5(dataset_dir, dataset_hdf5, 12, end=45, MEBCRN=True)
acqs = acqs[:,:,::4,::4,:]
out_maps = out_maps[:,::4,::4,:]

# Pre-process out maps
out_rho_w = np.expand_dims(out_maps[:,:,:,:2],axis=1)
out_rho_f = np.expand_dims(out_maps[:,:,:,2:4],axis=1)
out_xi = np.expand_dims(out_maps[:,:,:,4:],axis=1)
out_maps = np.concatenate((out_rho_w,out_rho_f,out_xi),axis=1)

################################################################################
########################### DATASET PARTITIONS #################################
################################################################################

# Overall dataset statistics
len_dataset,ne,hgt,wdt,n_ch = np.shape(acqs)
_,n_out,_,_,_ = np.shape(out_maps)

print('Dataset size:', len_dataset)
print('Acquisition Dimensions:', hgt,wdt)
print('Output Maps:',n_out)

A_B_dataset = tf.data.Dataset.from_tensor_slices((acqs,out_maps))
A_B_dataset = A_B_dataset.batch(1).shuffle(len_dataset)

enc= dl.encoder(input_shape=(6,hgt,wdt,n_ch),
                encoded_dims=64,
                filters=12,
                ls_reg_weight=1e-6,
                NL_self_attention=False)
dec= dl.decoder(encoded_dims=64,
                output_2D_shape=(hgt,wdt),
                filters=12,
                NL_self_attention=False)
G_A2B = keras.Sequential()
G_A2B.add(enc)
G_A2B.add(dec)

D_A = dl.PatchGAN(input_shape=(ne,hgt,wdt,2), dim=12, self_attention=False)

IDEAL_op = wf.IDEAL_Layer(ne,MEBCRN=True)

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn('wgan')
cycle_loss_fn = tf.losses.MeanSquaredError()

G_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
D_optimizer = keras.optimizers.Adam(learning_rate=8e-4, beta_1=0.5, beta_2=0.9)

A2B2A_pool = data.ItemPool(10)

@tf.function
def train_G(A, B):
    indices =tf.concat([tf.zeros((A.shape[0],1,hgt,wdt,2),dtype=tf.int32),
                        tf.ones((A.shape[0],1,hgt,wdt,2),dtype=tf.int32),
                        2*tf.ones((A.shape[0],1,hgt,wdt,2),dtype=tf.int32)],axis=1)
    PM_idx = tf.concat([tf.zeros_like(B[:,:,:,:1],dtype=tf.int32),
                        tf.ones_like(B[:,:,:,:1],dtype=tf.int32)],axis=-1)
    
    with tf.GradientTape() as t:
        ##################### A Cycle #####################
        A2B = G_A2B(A, training=True)
        A2B2A = IDEAL_op(A2B)

        ##################### B Cycle #####################
        B2A = IDEAL_op(B, training=False)
        B2A2B = G_A2B(B2A, training=True)

        ############## Discriminative Losses ##############
        A2B2A_d_logits = D_A(A2B2A, training=False)
        A2B2A_g_loss = g_loss_fn(A2B2A_d_logits)

        ############ Cycle-Consistency Losses #############
        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)

        ################ Regularizers #####################
        activ_reg = tf.add_n(G_A2B.losses)
        
        G_loss = activ_reg + 1e0*(A2B2A_cycle_loss + B2A2B_cycle_loss) + A2B2A_g_loss
        
    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables))

    return A2B2A,  {'A2B2A_g_loss': A2B2A_g_loss,
                    'A2B2A_cycle_loss': A2B2A_cycle_loss,
                    'B2A2B_cycle_loss': B2A2B_cycle_loss,
                    'LS_reg': activ_reg}


@tf.function
def train_D(A, A2B2A):
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
    A2B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B2A = A2B2A_pool(A2B2A)
    for _ in range(5):
        D_loss_dict = train_D(A, A2B2A)

    return G_loss_dict, D_loss_dict


# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join('output', 'test-grad'))

# main loop
for ep in range(10):
    for A, B in tqdm.tqdm(A_B_dataset, desc='Ep. '+str(ep+1), total=len_dataset):
        # ==============================================================================
        # =                                RANDOM TEs                                  =
        # ==============================================================================
        
        G_loss_dict, D_loss_dict = train_step(A, B)
        
        # summary
        with train_summary_writer.as_default():
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')