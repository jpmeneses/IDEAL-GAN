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

def acq_add_te(acqs,out_maps,te_orig,te_mod):
    n_batch,hgt,wdt,d_ech = acqs.shape
    n_ech = d_ech//2

    ne = te_mod.shape[1]
    dte = te_mod - te_orig
    all_dte = tf.reshape(dte,[-1])

    # Generate complex signal
    real_S = acqs[:,:,:,0::2]
    imag_S = acqs[:,:,:,1::2]
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((n_batch,hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(S, [num_voxel, ne]) # shape: (bs*nv,ne)

    # Split water/fat images (orig_rho) and param. maps
    orig_rho = out_maps[:,:,:,:4]
    param_maps = out_maps[:,:,:,4:]
    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    
    # Generate complex water/fat signals
    real_rho = orig_rho[:,:,:,0::2]
    imag_rho = orig_rho[:,:,:,1::2]
    rho = tf.complex(real_rho,imag_rho) * rho_sc 

    rho_mtx = tf.reshape(rho, [num_voxel, ns])
    rho_mtx = tf.expand_dims(rho_mtx,-1) # shape: (bs*nv,ns,1)

    Arho = tf.linalg.matmul(A_p,rho_mtx) # shape: (bs*nv,np,1)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[-1]) # shape: (bs*nv)

    D_xi = 2j*np.pi*xi_rav # shape: (bs*nv)
    D_p = tf.linalg.diag(2j*np.pi*tf.squeeze(f_p)) # shape: (np,np)

    def ode_fn(t,y):
        W = tf.math.exp(xi_rav*tf.cast(t,tf.complex64)) # shape: (bs*nv)
        P = tf.math.exp(2j*np.pi*f_p*tf.cast(t,tf.complex64)) # shape: (1,np)
        DpArho = tf.linalg.matmul(D_p,Arho) # shape: (bs*nv,np,1)
        PDpArho = tf.linalg.matmul(P,DpArho) # shape: (bs*nv,1,1)
        WPDpArho = W*tf.squeeze(PDpArho, axis=[1,2]) # shape: (bs*nv)
        return D_xi*y + WPDpArho

    def jacobian_fn(t,y):
        return tf.linalg.diag(D_xi)
        # return tf.concat([D_xi, tf.ones(D_xi.shape,dtype=tf.complex64)],axis=0)

    # Solve for first echo only
    # res = tfp.math.ode.BDF().solve(ode_fn, te_orig[0,0], Smtx[:,0], solution_times=[all_dte[0]], jacobian_fn=jacobian_fn)
    Smtx_aux = tf.transpose(res.states)
    Smtx_hat = tf.zeros([n_batch*num_voxel, ne-1], dtype=tf.complex64)
    Smtx_hat = tf.concat([Smtx_aux, Smtx_hat], axis=-1)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(Smtx_hat,[n_batch,hgt,wdt,ne])

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