import tensorflow as tf
import numpy as np

# Multipeak fat model
species = ["water", "fat"]
ns = len(species)

field = 1.5

f_p = np.array([ 0., -3.80, -3.40, -2.60, -1.94, -0.39, 0.60 ]) * 1E-6 * 42.58E6 * field
f_p = tf.convert_to_tensor(f_p,dtype=tf.complex64)
f_p = tf.expand_dims(f_p,0)

A_p = np.array([[1.0,0.0],[0.0,0.087],[0.0,0.693],[0.0,0.128],[0.0,0.004],[0.0,0.039],[0.0,0.048]])
A_p = tf.convert_to_tensor(A_p,dtype=tf.complex64)

r2_sc = 200.0   # HR:150 / GC:200
fm_sc = 300.0   # HR:300 / GC:400
rho_sc = 1.4

def gen_M(te,get_Mpinv=True,get_P0=False):
    ne = te.shape[1] # len(te)
    te = tf.cast(te,tf.complex64)
    te = tf.expand_dims(te,-1)

    M = tf.linalg.matmul(tf.math.exp(tf.tensordot(2j*np.pi*te,f_p,axes=1)),A_p) # shape: bs x ne x ns

    Q, R = tf.linalg.qr(M)
    if get_P0:
        P0 = tf.eye(ne,dtype=tf.complex64) - tf.linalg.matmul(Q, tf.transpose(Q,perm=[0,2,1],conjugate=True))
        P0 = 0.5 * (tf.transpose(P0,perm=[0,2,1],conjugate=True) + P0)

    # Pseudo-inverse
    if get_Mpinv:
        M_pinv = tf.linalg.solve(R, tf.transpose(Q,perm=[0,2,1],conjugate=True))

    if get_P0 and get_Mpinv:
        return M, P0, M_pinv
    elif get_Mpinv and not(get_P0):
        return M, M_pinv
    elif not(get_Mpinv) and not(get_P0):
        return M


def acq_to_acq(acqs,param_maps,te=None,complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    if complex_data:
        n_ech = d_ech
    else:
        n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
        te = tf.expand_dims(te,0)
        te = tf.tile(te,[n_batch,1])
    
    ne = te.shape[1]
    M, M_pinv = gen_M(te) # M shape: (bs,ne,ns)

    te_complex = tf.complex(0.0,te) # shape was: ne // now: (bs,ne)
    te_complex = tf.expand_dims(te_complex,-1) # shape: (bs,ne,1)

    # Generate complex signal
    if not(complex_data):
        real_S = acqs[:,:,:,0::2]
        imag_S = acqs[:,:,:,1::2]
        S = tf.complex(real_S,imag_S)
    else:
        S = acqs

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1]) # shape: (bs,nv,ne)

    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    # r2s = tf.zeros_like(phi)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1]) # shape: (bs,nv)
    xi_rav = tf.expand_dims(xi_rav,1) # shape: (bs,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (bs,ne,nv)
    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))

    # Matrix operations
    WmS = Wm * Smtx # shape = (bs,ne,nv)
    MWmS = tf.linalg.matmul(M_pinv,WmS) # shape = (bs,ns,nv)
    MMWmS = tf.linalg.matmul(M,MWmS) # shape = (bs,ne,nv)
    Smtx_hat = Wp * MMWmS # shape = (bs,ne,nv)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(tf.transpose(MWmS, perm=[0,2,1]),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    zero_fill = tf.zeros_like(Re_rho)
    re_stack = tf.stack([Re_rho,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ns])
    im_stack = tf.stack([zero_fill,Im_rho],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ns])
    res_rho = re_aux + im_aux

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx_hat, perm=[0,2,1]),[n_batch,hgt,wdt,ne])

    if not(complex_data):
        Re_gt = tf.math.real(S_hat)
        Im_gt = tf.math.imag(S_hat)
        zero_fill = tf.zeros_like(Re_gt)
        re_stack = tf.stack([Re_gt,zero_fill],4)
        re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*n_ech])
        im_stack = tf.stack([zero_fill,Im_gt],4)
        im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*n_ech])
        res_gt = re_aux + im_aux
        return (res_rho,res_gt)
    else:
        return (res_rho,S_hat)


def IDEAL_model(out_maps,n_ech,te=None,complex_data=False):
    n_batch,hgt,wdt,_ = out_maps.shape

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
        te = tf.expand_dims(te,0)
        te = tf.tile(te,[n_batch,1])

    ne = te.shape[1]
    M = gen_M(te,get_Mpinv=False)

    te_complex = tf.complex(0.0,te)
    te_complex = tf.expand_dims(te_complex,-1)

    # Split water/fat images (orig_rho) and param. maps
    orig_rho = out_maps[:,:,:,:4]
    param_maps = out_maps[:,:,:,4:]

    # Generate complex water/fat signals
    real_rho = orig_rho[:,:,:,0::2]
    imag_rho = orig_rho[:,:,:,1::2]
    rho = tf.complex(real_rho,imag_rho) * rho_sc

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.transpose(tf.reshape(rho, [n_batch, num_voxel, ns]), perm=[0,2,1])

    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1)

    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))

    # Matrix operations
    Smtx = Wp * tf.linalg.matmul(M,rho_mtx)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx, perm=[0,2,1]),[n_batch,hgt,wdt,ne])

    if not(complex_data):
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
    else:
        return S_hat


def get_Ps_norm(acqs,param_maps,te=None):
    n_batch,hgt,wdt,d_ech = acqs.shape
    n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
        te = tf.expand_dims(te,0)
        te = tf.tile(te,[n_batch,1])

    ne = te.shape[1]
    M, P0, M_pinv = gen_M(te,get_P0=True)

    te_complex = tf.complex(0.0,te)
    te_complex = tf.expand_dims(te_complex,-1)

    # Generate complex signal
    real_S = acqs[:,:,:,0::2]
    imag_S = acqs[:,:,:,1::2]
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1])

    # r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    r2s = tf.zeros_like(phi)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav))
    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))

    # Matrix operations
    WmS = Wm * Smtx
    PWmS = tf.linalg.matmul(P0,WmS)
    T = Wp * PWmS
    
    # Reshape to original images dimensions
    Ps = tf.reshape(tf.transpose(T,perm=[0,2,1]),[n_batch,hgt,wdt,ne])
    
    # L2 norm
    L2_norm_vec = tf.math.reduce_euclidean_norm(Ps,axis=[-3,-2])
    # L2_norm = tf.abs(tf.reduce_mean(tf.reduce_sum(L2_norm_vec,axis=-1)))
    L2_norm = tf.abs(tf.reduce_sum(L2_norm_vec))

    return L2_norm


def get_rho(acqs,param_maps,te=None,complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    if complex_data:
        n_ech = d_ech
    else:
        n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
        te = tf.expand_dims(te,0)
        te = tf.tile(te,[n_batch,1])

    ne = te.shape[1]
    M, M_pinv = gen_M(te)

    te_complex = tf.complex(0.0,te)
    te_complex = tf.expand_dims(te_complex,-1)

    # Generate complex signal
    if not(complex_data):
        real_S = acqs[:,:,:,0::2]
        imag_S = acqs[:,:,:,1::2]
        S = tf.complex(real_S,imag_S)
    else:
        S = acqs

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1])

    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav))

    # Matrix operations
    WmS = Wm * Smtx
    MWmS = tf.linalg.matmul(M_pinv,WmS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(tf.transpose(MWmS, perm=[0,2,1]),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    zero_fill = tf.zeros_like(Re_rho)
    re_stack = tf.stack([Re_rho,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ns])
    im_stack = tf.stack([zero_fill,Im_rho],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ns])
    res_rho = re_aux + im_aux
    
    return res_rho


def PDFF_uncertainty(acqs, mean_maps, var_maps, te=None, complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    if complex_data:
        n_ech = d_ech
    else:
        n_ech = d_ech//2

    if te is None:
        stop_te = (n_ech*12/6)*1e-3
        te = np.arange(start=1.3e-3,stop=stop_te,step=2.1e-3)
        te = tf.convert_to_tensor(te,dtype=tf.float32)
        te = tf.expand_dims(te,0)
        te = tf.tile(te,[n_batch,1])

    ne = te.shape[1]
    M, M_pinv = gen_M(te)

    te_complex = tf.complex(0.0,te)
    te_complex = tf.expand_dims(te_complex,-1)
    te_real = tf.complex(te,0.0)
    te_real = tf.expand_dims(te_real, -1)

    # Generate complex signal
    if not(complex_data):
        real_S = acqs[:,:,:,0::2]
        imag_S = acqs[:,:,:,1::2]
        S = tf.complex(real_S,imag_S)
    else:
        S = acqs

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1])

    # r2s = mean_maps[:,:,:,0] * r2_sc
    phi = mean_maps * fm_sc
    # r2s_unc = param_maps[:,:,:,0] * r2_sc
    phi_unc = var_maps * (fm_sc**2)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,0.0)
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1)
    xi_unc = tf.complex(phi_unc,0.0)
    xi_unc_rav = tf.reshape(xi_unc,[n_batch,-1])
    xi_unc_rav = tf.expand_dims(xi_unc_rav,1)

    # Diagonal matrix with the exponential of fieldmap variance
    Wm_var = tf.math.exp(tf.linalg.matmul(-2*(2*np.pi * te_real)**2, xi_unc_rav))

    # (New) Diagonal matrix with sine and cosine terms
    # (derived from variance of complex exponential)
    # https://nbviewer.org/gist/dougalsutherland/8513749
    # Z_sin = tf.math.sin(tf.linalg.matmul(2*np.pi * te_real, xi_rav))
    Z_cos = tf.math.cos(tf.linalg.matmul(4*np.pi * te_real, xi_rav))
    # Z = Z_sin**2 - Z_cos**2

    # Matrix operations
    WmZS = (1 + Wm_var * Z_cos) * (Smtx * tf.math.conj(Smtx))
    MWmZS = tf.linalg.matmul(M_pinv * tf.math.conj(M_pinv),WmZS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_var = tf.reshape(tf.transpose(MWmZS, perm=[0,2,1]),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho = tf.math.real(rho_var)
    Im_rho = tf.math.imag(rho_var)
    zero_fill = tf.zeros_like(Re_rho)
    re_stack = tf.stack([Re_rho,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*ns])
    im_stack = tf.stack([zero_fill,Im_rho],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*ns])
    res_rho = re_aux + im_aux
    
    return res_rho