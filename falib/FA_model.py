import tensorflow as tf
import numpy as np

# Multipeak fat model
species = ["water", "fat", "ndb", "nmidb", "cl"]
ns = len(species)

field = 1.5

# f_p = np.array([ 0., 0.47, -0.64, -2.09, -2.60, -2.82, -3.23, -3.54, 3.95 ]) * 1E-6 * 42.58E6 * field
f_p = np.array([ 0., 0.72, -0.43, -1.82, -2.33, -2.57, -2.97, -3.29, 3.70 ]) * 1E-6 * 42.58E6 * field
f_p = tf.convert_to_tensor(f_p,dtype=tf.complex64)
f_p = tf.expand_dims(f_p,0)

t2_p = np.array([ 2000.0, 46.5, 30.6, 48.4, 47.2, 36.3, 37.6, 82.6, 79.4 ]) * 1E-3
t2_p = tf.convert_to_tensor(t2_p,dtype=tf.complex64)
t2_p = tf.expand_dims(t2_p,0)

A_p = np.array([[1,  0, 0, 0,0],
                [0,  1, 2, 0,0],
                [0,  4, 0, 0,0],
                [0,  0, 0, 2,0],
                [0,  6, 0, 0,0],
                [0,  0, 4,-4,0],
                [0,  0, 0, 0,0],
                [0,-24,-8, 2,6],
                [0,  9, 0, 0,0]])
A_p = tf.convert_to_tensor(A_p,dtype=tf.complex64)

r2_sc = 200.0   # HR:150 / GC:200
fm_sc = 300.0   # HR:300 / GC:400
rho_sc = 1.4


def gen_M(te,get_Mpinv=True,get_P0=False):
    ne = len(te)
    te = tf.cast(te,tf.complex64)
    te = tf.expand_dims(te,-1)

    M = tf.linalg.matmul(tf.math.exp(tf.tensordot(te,-1/t2_p+2j*np.pi*f_p,axes=1)),A_p)

    Q, R = tf.linalg.qr(M)
    if get_P0:
        P0 = tf.eye(ne,dtype=tf.complex64) - tf.linalg.matmul(Q, tf.transpose(Q,conjugate=True))
        P0 = 0.5 * (tf.transpose(P0,conjugate=True) + P0)

    # Pseudo-inverse
    if get_Mpinv:
        M_pinv = tf.linalg.solve(R, tf.transpose(Q,conjugate=True))

    if get_P0 and get_Mpinv:
        return M, P0, M_pinv
    elif get_Mpinv and not(get_P0):
        return M, M_pinv
    elif not(get_Mpinv) and not(get_P0):
        return M


def acq_to_acq(acqs,param_maps,te=None,complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
    else:
        # te: TF array with the echo times - shape: (n_batch,ne)
        te = tf.squeeze(te,[0])
    
    ne = len(te)
    M, M_pinv = gen_M(te)

    te_complex = tf.complex(0.0,te)

    # Generate complex signal
    real_S = acqs[:,:,:,0::2]
    imag_S = acqs[:,:,:,1::2]
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((n_batch,hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [num_voxel, ne]))

    # r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    r2s = tf.zeros_like(phi)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[-1])

    Wm = tf.math.exp(tf.tensordot(-2*np.pi * te_complex, xi_rav, axes=0))
    Wp = tf.math.exp(tf.tensordot(+2*np.pi * te_complex, xi_rav, axes=0))

    # Matrix operations
    WmS = Wm * Smtx
    MWmS = tf.linalg.matmul(M_pinv,WmS)
    MMWmS = tf.linalg.matmul(M,MWmS)
    Smtx_hat = Wp * MMWmS

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(tf.transpose(MWmS),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    zero_fill = tf.zeros_like(Re_rho)
    re_stack = tf.stack([Re_rho,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ns])
    im_stack = tf.stack([zero_fill,Im_rho],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ns])
    res_rho = re_aux + im_aux

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx_hat),[n_batch,hgt,wdt,ne])

    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    zero_fill = tf.zeros_like(Re_gt)
    re_stack = tf.stack([Re_gt,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*n_ech])
    im_stack = tf.stack([zero_fill,Im_gt],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*n_ech])
    res_gt = re_aux + im_aux
    
    return (res_rho,res_gt)


def IDEAL_model(out_maps,n_ech,te=None,complex_data=False):
    n_batch,hgt,wdt,_ = out_maps.shape

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
    else:
        # te: TF array with the echo times - shape: (n_batch,ne)
        te = tf.squeeze(te,[0])

    ne = len(te)
    M = gen_M(te,get_Mpinv=False)

    te_complex = tf.complex(0.0,te)

    # Split water/fat images (orig_rho) and param. maps
    orig_rho = out_maps[:,:,:,:4]
    param_maps = out_maps[:,:,:,4:]

    # Generate complex water/fat signals
    real_rho = orig_rho[:,:,:,0::2]
    imag_rho = orig_rho[:,:,:,1::2]
    rho = tf.complex(real_rho,imag_rho) * rho_sc

    voxel_shape = tf.convert_to_tensor((n_batch,hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.transpose(tf.reshape(rho, [num_voxel, ns]))

    # r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    r2s = tf.zeros_like(phi)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[-1])

    Wp = tf.math.exp(tf.tensordot(+2*np.pi * te_complex, xi_rav, axes=0))

    # Matrix operations
    Smtx = Wp * tf.linalg.matmul(M,rho_mtx)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx),[n_batch,hgt,wdt,ne])

    # Split into real and imaginary channels
    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    zero_fill = tf.zeros_like(Re_gt)
    re_stack = tf.stack([Re_gt,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ne])
    im_stack = tf.stack([zero_fill,Im_gt],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ne])
    res_gt = re_aux + im_aux
    
    return res_gt


def get_rho(acqs,param_maps,te=None,complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
    else:
        # te: TF array with the echo times - shape: (n_batch,ne)
        te = tf.squeeze(te,[0])

    ne = len(te)
    M, M_pinv = gen_M(te)

    te_complex = tf.complex(0.0,te)

    # Generate complex signal
    real_S = acqs[:,:,:,0::2]
    imag_S = acqs[:,:,:,1::2]
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((n_batch,hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    voxel_ns_shape = voxel_shape + (ns,)
    Smtx = tf.transpose(tf.reshape(S, [num_voxel, ne]))

    # r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    r2s = tf.zeros_like(phi)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[-1])

    Wm = tf.math.exp(tf.tensordot(-2*np.pi * te_complex, xi_rav, axes=0))

    # Matrix operations
    WmS = Wm * Smtx
    MWmS = tf.linalg.matmul(M_pinv,WmS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(tf.transpose(MWmS),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    zero_fill = tf.zeros_like(Re_rho)
    re_stack = tf.stack([Re_rho,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ns])
    im_stack = tf.stack([zero_fill,Im_rho],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ns])
    res_rho = re_aux + im_aux
    
    return res_rho