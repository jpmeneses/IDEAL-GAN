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

def acq_add_te(acqs,out_maps,te_orig,te_mod,complex_data=False):
    n_batch,hgt,wdt,d_ech = acqs.shape
    if complex_data:
        n_ech = d_ech
    else:
        n_ech = d_ech//2

    ne = te_orig.shape[1]

    # Generate complex signal
    if not(complex_data):
        real_S = acqs[:,:,:,0::2]
        imag_S = acqs[:,:,:,1::2]
        S = tf.complex(real_S,imag_S)
    else:
        S = acqs

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1]) # shape: (bs,ne,nv)

    # Split water/fat images (orig_rho) and param. maps
    orig_rho = out_maps[:,:,:,:4]
    param_maps = out_maps[:,:,:,4:]
    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    
    # Generate complex water/fat signals
    real_rho = orig_rho[:,:,:,0::2]
    imag_rho = orig_rho[:,:,:,1::2]
    rho = tf.complex(real_rho,imag_rho) * rho_sc 

    rho_mtx = tf.transpose(tf.reshape(rho, [n_batch, num_voxel, ns]), perm=[0,2,1]) # shape: (bs,ns,nv)

    Arho = tf.linalg.matmul(A_p,rho_mtx) # shape: (bs,np,nv)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch, num_voxel]) # shape: (bs,nv)

    D_xi = 2j*np.pi*xi_rav # shape: (bs,nv)
    D_xi = tf.expand_dims(D_xi,1) # shape: (bs,1,nv)
    D_p = tf.linalg.diag(2j*np.pi*tf.squeeze(f_p)) # shape: (np,np)
    DpArho = tf.linalg.matmul(D_p,Arho) # shape: (bs,np,nv)

    dt = 0.00001e-3

    for ech_idx in range(ne):
        S_ech = Smtx[:,ech_idx,:] # shape: (bs,nv)
        S_ech = tf.expand_dims(S_ech,1) # shape: (bs,1,nv)
        te_ini = tf.cast(tf.squeeze(te_orig[:,ech_idx]),tf.float32)
        te_end = tf.cast(tf.squeeze(te_mod[:,ech_idx]),tf.float32)
        if te_ini <= te_end:
            t_vec = tf.range(start=te_ini,limit=te_end,delta=dt)
        else:
            t_vec = tf.range(start=te_end,limit=te_ini,delta=dt)
        for t in t_vec:
            t = tf.cast(t,tf.complex64)
            P_ech = tf.math.exp(2j*np.pi*t*f_p) # shape: (1,np)
            W_ech = tf.math.exp(xi_rav*t) # shape: (bs,nv)
            W_ech = tf.expand_dims(W_ech,1) # shape: (bs,1,nv)
            PDpArho = tf.linalg.matmul(P_ech,DpArho) # shape: (bs,1,nv)
            WPDpArho = W_ech * PDpArho # shape: (bs,1,nv)
            S_dt = D_xi*S_ech + WPDpArho
            S_ech += S_dt*dt
        if ech_idx == 0:
            Smtx_hat = S_ech
        else:
            Smtx_hat = tf.concat([Smtx_hat, S_ech], axis=1)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx_hat, perm=[0,2,1]),[n_batch,hgt,wdt,ne])

    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    zero_fill = tf.zeros_like(Re_gt)
    re_stack = tf.stack([Re_gt,zero_fill],4)
    re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*n_ech])
    im_stack = tf.stack([zero_fill,Im_gt],4)
    im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*n_ech])
    res_gt = re_aux + im_aux
    
    return res_gt


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

    ne = len(te)
    M, M_pinv, P0 = gen_M(te,get_P0=True)

    te_complex = tf.complex(0.0,te)
    te_complex = tf.expand_dims(te_complex,-1)

    # Generate complex signal
    real_S = acqs[:,:,:,0::2]
    imag_S = acqs[:,:,:,1::2]
    S = tf.complex(real_S,imag_S)

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

def abs_acq_to_acq(acqs,param_maps,te=None,complex_data=False):
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

    te = tf.expand_dims(te,-1) # shape: (bs,ne,1)
    te_complex = tf.complex(0.0,te) # shape: (bs,ne,1)

    # Generate complex signal
    if not(complex_data):
        real_S = acqs[:,:,:,0::2]
        imag_S = acqs[:,:,:,1::2]
        S = tf.complex(real_S,imag_S)
    else:
        S = acqs

    S_abs = tf.abs(S)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1]) # shape: (bs,nv,ne)

    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1]) # shape: (bs,nv)
    xi_rav = tf.expand_dims(xi_rav,1) # shape: (bs,1,nv)

    r2s_rav = tf.reshape(r2s,[n_batch,-1]) # shape: (bs,nv)
    r2s_rav = tf.expand_dims(r2s_rav,1) # shape: (bs,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (bs,ne,nv)
    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))
    Wp_abs = tf.math.exp(tf.linalg.matmul(te, -r2s_rav))

    # Matrix operations
    WmS = Wm * Smtx # shape = (bs,ne,nv)
    MWmS = tf.linalg.matmul(M_pinv,WmS) # shape = (bs,ns,nv)
    MMWmS = tf.linalg.matmul(M,MWmS) # shape = (bs,ne,nv)
    Smtx_hat_abs = Wp_abs * tf.abs(MMWmS) # shape = (bs,ne,nv)

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
    S_hat_abs = tf.reshape(tf.transpose(Smtx_hat_abs, perm=[0,2,1]),[n_batch,hgt,wdt,ne])

    return (res_rho,S_hat_abs)