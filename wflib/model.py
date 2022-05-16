import tensorflow as tf
import numpy as np

r2_sc = 200.0   # HR:150 / GC:200
fm_sc = 300.0   # HR:300 / GC:400
pi = tf.constant(np.pi,dtype=tf.complex64)

# Echo Times - Dim 1x6
# TE_ini = tf.range(start=1.3e-3,limit=12e-3,delta=2.1e-3,dtype=tf.float32)
# n_ech = len(TE_ini)
# TE = tf.complex(TE_ini,0.0)

# Fat Peaks Constants - Dim 7x1
f_peaks = tf.constant([[0.0],[-3.80],[-3.40],[-2.60],[-1.94],[-0.39],[0.60]],dtype=tf.complex64)*1e-6*42.58e6*1.5
a_peaks = tf.constant([[0.0],[0.087],[0.693],[0.128],[0.004],[0.039],[0.048]],dtype=tf.complex64)
    
def signal_model(y_actual,TE):
    n_batch,hgt,wdt,_ = y_actual.shape
    TE = TE[0]
    n_ech = len(TE)
    TE = tf.complex(TE,0.0)

    # Sum along the fat peaks' values
    fact_exp = tf.linalg.matmul(f_peaks,tf.expand_dims(TE,0)) # Dim 7x6
    f_term = tf.math.reduce_sum(a_peaks*tf.math.exp(2j*pi*fact_exp),axis=0) # Dim 1x6
    
    aux = tf.tile(tf.expand_dims(TE,1),[1,wdt])
    aux2 = tf.linalg.diag(aux)
    TE_mat = tf.transpose(aux2,perm=[2,1,0])
    
    aux3 = tf.tile(tf.expand_dims(f_term,1),[1,wdt])
    aux4 = tf.linalg.diag(aux3)
    ft_mat = tf.transpose(aux4,perm=[2,1,0])
    
    # Turn Maps Values to Complex
    y_act_comp = tf.complex(y_actual,0.0)
    
    # Signal for ground-truth value
    W_gt = tf.complex(y_actual[:,:,:,0],y_actual[:,:,:,1])*1.4
    W_gt = tf.tile(tf.expand_dims(W_gt,-1),[1,1,1,n_ech])
    F_gt = tf.complex(y_actual[:,:,:,2],y_actual[:,:,:,3])*1.4
    F_gt = tf.tile(tf.expand_dims(F_gt,-1),[1,1,1,wdt])
    r2_orig = y_act_comp[:,:,:,4]
    r2_gt = tf.tile(tf.expand_dims(r2_orig,-1),[1,1,1,wdt])
    fm_orig = y_act_comp[:,:,:,5]
    fm_gt = tf.tile(tf.expand_dims(fm_orig,-1),[1,1,1,wdt])
    gt_1 = tf.math.exp(-r2_sc*tf.linalg.matmul(r2_gt,TE_mat))         # Dim 1x6
    gt_2 = tf.math.exp(2j*pi*fm_sc*tf.linalg.matmul(fm_gt,TE_mat))   # Dim 1x6
    gt_3 = (W_gt + tf.linalg.matmul(F_gt,ft_mat))                     # Dim 1x6
    In_gt =  gt_1 * gt_2 * gt_3
    
    # Post-process model reconstructed acquisitions
    Re_gt = tf.math.real(In_gt)
    Im_gt = tf.math.imag(In_gt)
    zero_fill = tf.zeros_like(Re_gt)
    re_stack = tf.stack([Re_gt,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*n_ech])
    im_stack = tf.stack([zero_fill,Im_gt],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*n_ech])
    res_gt = re_aux + im_aux
    
    return res_gt