import tensorflow as tf
import numpy as np

def gen_TEvar(n_ech,bs,orig=False):
    stp_te = (12.5*n_ech/6)*1e-3
    if orig:
        TE_ini_var = 1.3 * 1e-3
        d_TE_var = 2.1 * 1e-3
    else:
        TE_ini_var = (1.2 + 0.2*np.random.uniform()) * 1e-3
        d_TE_var = (1.9 + 0.3*np.random.uniform()) * 1e-3
    te_var_np = np.arange(start=TE_ini_var,stop=stp_te,step=d_TE_var)
    te_var = tf.convert_to_tensor(te_var_np,dtype=tf.float32)
    te_var = tf.expand_dims(te_var,0)
    te_var = tf.tile(te_var,[bs,1])
    return te_var