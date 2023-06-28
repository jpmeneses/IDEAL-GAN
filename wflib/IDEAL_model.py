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


@tf.function
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


@tf.function
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
        zero_fill = tf.zeros_like(Re_gt,dtype=tf.float32)
        re_stack = tf.stack([Re_gt,zero_fill],4)
        re_aux = tf.reshape(re_stack,[n_batch,hgt,wdt,2*n_ech])
        im_stack = tf.stack([zero_fill,Im_gt],4)
        im_aux = tf.reshape(im_stack,[n_batch,hgt,wdt,2*n_ech])
        res_gt = re_aux + im_aux
        return (res_rho,res_gt)
    else:
        return (res_rho,S_hat)


@tf.custom_gradient
def IDEAL_model(out_maps):
    n_batch,_,hgt,wdt,_ = out_maps.shape
    ne = 6
    
    te = np.arange(start=1.3e-3,stop=12*1e-3,step=2.1e-3)
    te = tf.expand_dims(tf.convert_to_tensor(te,dtype=tf.float32),0) # (1,ne)
    te_complex = tf.complex(0.0,te) # (1,ne)
    
    M = gen_M(te,get_Mpinv=False)

    # Generate complex water/fat signals
    real_rho = tf.transpose(out_maps[:,:2,:,:,0],perm=[0,2,3,1])
    imag_rho = tf.transpose(out_maps[:,:2,:,:,1],perm=[0,2,3,1])
    rho = tf.complex(real_rho,imag_rho) * rho_sc

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.transpose(tf.reshape(rho, [n_batch, num_voxel, ns]), perm=[0,2,1])

    r2s = (out_maps[:,2,:,:,0]*0.5 + 0.5) * r2_sc 
    phi = out_maps[:,2,:,:,1] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,-1) # (nb,nv,1)

    Wp = tf.math.exp(tf.linalg.matmul(xi_rav, +2*np.pi * te_complex)) # (nb,nv,ne)
    Wp = tf.transpose(Wp, perm=[0,2,1]) # (nb,ne,nv)

    # Matrix operations
    Smtx = Wp * tf.linalg.matmul(M,rho_mtx) # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(tf.transpose(Smtx, perm=[0,2,1]),[n_batch,hgt,wdt,ne])
    S_hat = tf.expand_dims(tf.transpose(S_hat, perm=[0,3,1,2]),-1)
    
    # Split into real and imaginary channels
    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    res_gt = tf.concat([Re_gt,Im_gt],axis=-1)
    
    def grad(upstream): # Must be same shape as out_maps
        # Water/fat gradient
        Wp_d = tf.linalg.diag(tf.transpose(Wp,perm=[0,2,1])) # (nb,nv,ne,ne)
        ds_dp = tf.linalg.matmul(Wp_d,M) * rho_sc ## (nb,nv,ne,ns) I1
        
        # Xi gradient, considering Taylor approximation
        dxi = tf.squeeze(tf.linalg.diag(2*np.pi*te_complex)) # (1,ne) --> (ne,ne)
        ds_dxi = tf.linalg.matmul(dxi,Smtx) # (nb,ne,nv)
        ds_dxi = tf.expand_dims(tf.transpose(ds_dxi,perm=[0,2,1]),axis=-1) ## (nb,nv,ne,1) I2

        # Concatenate d_s/d_param gradients
        ds_dq = tf.concat([ds_dp,ds_dxi],axis=-1) # (nb,nv,ne,3)
        ds_dq = tf.transpose(ds_dq, perm=[0,1,3,2]) * fm_sc ## (nv,nv,3,ne)

        # Re-format upstream 
        upstream = tf.complex(upstream[:,:,:,:,0],upstream[:,:,:,:,1]) # (nb,ne,hgt,wdt)
        upstream = tf.transpose(tf.reshape(upstream, [n_batch,ne,num_voxel]), perm=[0,2,1]) # (nb,nv,ne)

        grad_res = tf.linalg.matvec(ds_dq, upstream) # (nb,nv,3)
        grad_res = tf.reshape(tf.transpose(grad_res,perm=[0,2,1]), [n_batch,ns+1,hgt,wdt]) # (nb,3,hgt,wdt)
        grad_res_r = tf.math.real(tf.expand_dims(grad_res,axis=-1))
        grad_res_i = tf.math.imag(tf.expand_dims(grad_res,axis=-1))
        grad_res = tf.concat([grad_res_r,grad_res_i],axis=-1) # (nb,3,hgt,wdt,2)

        return grad_res
    
    return res_gt, grad


class IDEAL_Layer(tf.keras.layers.Layer):
    def __init__(self,n_ech,MEBCRN=False):
        super(IDEAL_Layer, self).__init__()
        self.n_ech = n_ech
        self.MEBCRN = MEBCRN

    def call(self,out_maps,te=None,training=None):
        return IDEAL_model(out_maps)


@tf.function
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

    r2s = param_maps[:,:,:,0] * r2_sc
    phi = param_maps[:,:,:,1] * fm_sc
    # r2s = tf.zeros_like(phi)

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


@tf.function
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


@tf.function
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

    # te_complex = tf.expand_dims(tf.complex(0.0,te),-1)
    te_real = tf.expand_dims(tf.complex(te,0.0), -1)

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

    r2s = mean_maps[:,:,:,0] * r2_sc
    phi = mean_maps[:,:,:,1] * fm_sc
    r2s_unc = var_maps[:,:,:,0] * (r2_sc**2)
    phi_unc = var_maps[:,:,:,1] * (fm_sc**2)
    
    r2s_rav = tf.reshape(tf.complex(r2s,0.0),[n_batch,-1])
    r2s_rav = tf.expand_dims(r2s_rav,1)
    r2s_unc_rav = tf.reshape(tf.complex(r2s_unc,0.0),[n_batch,-1])
    r2s_unc_rav = tf.expand_dims(r2s_unc_rav,1)
    phi_unc_rav = tf.reshape(tf.complex(phi_unc,0.0),[n_batch,-1])
    phi_unc_rav = tf.expand_dims(phi_unc_rav,1)

    # Diagonal matrix with the exponential of fieldmap variance
    r2s_var_aux = tf.linalg.matmul(te_real**2, r2s_unc_rav)
    Wm_unc_r2s = tf.math.exp(tf.linalg.matmul(2*te_real, r2s_rav) + r2s_var_aux)
    Wm_var_r2s = tf.math.exp(r2s_var_aux)
    Wm_var_phi = tf.math.exp(tf.linalg.matmul(-(2*np.pi * te_real)**2, phi_unc_rav))
    Wm_var = -(1 - Wm_var_phi) * (1 - Wm_var_r2s) * Wm_unc_r2s

    # Matrix operations (variance)
    WmZS = Wm_var * (Smtx * tf.math.conj(Smtx))
    MWmZS = tf.linalg.matmul(M_pinv * tf.math.conj(M_pinv),WmZS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_var = tf.reshape(tf.transpose(MWmZS, perm=[0,2,1]),[n_batch,hgt,wdt,ns]) / rho_sc

    Re_rho_var = tf.math.real(rho_var)
    Im_rho_var = tf.math.imag(rho_var)
    zero_fill = tf.zeros_like(Re_rho_var)
    re_stack_var = tf.stack([Re_rho_var,zero_fill],4)
    re_aux_var = tf.reshape(re_stack_var,[n_batch,hgt,wdt,2*ns])
    im_stack_var = tf.stack([zero_fill,Im_rho_var],4)
    im_aux_var = tf.reshape(im_stack_var,[n_batch,hgt,wdt,2*ns])
    res_rho_var = re_aux_var + im_aux_var
    
    return res_rho_var