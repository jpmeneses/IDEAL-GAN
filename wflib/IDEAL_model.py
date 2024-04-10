import tensorflow as tf
import numpy as np

# Multipeak fat model
species = ["water", "fat"]
ns = len(species)

# field = 1.5

f_p = np.array([ 0., -3.80, -3.40, -2.60, -1.94, -0.39, 0.60 ]) * 1E-6 * 42.58E6 #* field
f_p = tf.convert_to_tensor(f_p,dtype=tf.complex64)
f_p = tf.expand_dims(f_p,0)

A_p = np.array([[1.0,0.0],[0.0,0.087],[0.0,0.693],[0.0,0.128],[0.0,0.004],[0.0,0.039],[0.0,0.048]])
A_p = tf.convert_to_tensor(A_p,dtype=tf.complex64)

r2_sc = 200.0   # HR:150 / GC:200
fm_sc = 300.0   # HR:300 / GC:400
rho_sc = 1.4

def gen_TEvar(n_ech, bs=1, orig=False, TE_ini_min=1.0e-3, TE_ini_d=1.4e-3, d_TE_min=1.6e-3, d_TE_d=1.0e-3):
    if orig:
        TE_ini_var = 1.3 * 1e-3
        d_TE_var = 2.1 * 1e-3
        stp_te = TE_ini_var + d_TE_var * (n_ech-1) + 1e-4
        te_var_np = np.arange(start=TE_ini_var,stop=stp_te,step=d_TE_var)
    elif not TE_ini_d and not d_TE_d:
        TE_ini_var = TE_ini_min
        d_TE_var = d_TE_min
        stp_te = TE_ini_var + d_TE_var * (n_ech-1) + 1e-4
        te_var_np = np.arange(start=TE_ini_var,stop=stp_te,step=d_TE_var)
    else:
        # TE_ini_var = (1.0 + 1.5*np.random.uniform()) * 1e-3
        TE_ini_var = TE_ini_min + np.random.uniform(0,TE_ini_d)
        # d_TE_var = (1.5 + 1.0*np.random.uniform()) * 1e-3
        d_TE_c = d_TE_min + np.random.uniform(0,d_TE_d)
        # The lowest the SD, the more equally-distanced TEs
        d_TE_var = np.random.normal(d_TE_c, 1e-4, size=(n_ech-1,))
        d_TE_var = np.concatenate((np.array([0.0]), d_TE_var), axis=0)
        te_var_np = np.cumsum(d_TE_var) + TE_ini_var
    te_var = tf.convert_to_tensor(te_var_np,dtype=tf.float32)
    te_var = tf.expand_dims(te_var,0)
    te_var = tf.tile(te_var,[bs,1])
    te_var = tf.expand_dims(te_var,-1)
    return te_var


@tf.function
def gen_M(te, field=1.5, get_Mpinv=True, get_P0=False):
    ne = te.shape[1] # len(te)
    te = tf.cast(te, tf.complex64)
    field = tf.cast(field, tf.complex64)

    M = tf.linalg.matmul(tf.math.exp(tf.linalg.matmul(2j*np.pi*te, field*f_p)), A_p) # shape: bs x ne x ns

    Q, R = tf.linalg.qr(M)
    if get_P0:
        P0 = tf.eye(ne, dtype=tf.complex64) - tf.linalg.matmul(Q, tf.transpose(Q,perm=[0,2,1], conjugate=True))
        P0 = 0.5 * (tf.transpose(P0, perm=[0,2,1], conjugate=True) + P0)

    # Pseudo-inverse
    if get_Mpinv:
        M_pinv = tf.linalg.solve(R, tf.transpose(Q, perm=[0,2,1], conjugate=True))

    if get_P0 and get_Mpinv:
        return M, P0, M_pinv
    elif get_Mpinv and not(get_P0):
        return M, M_pinv
    elif not(get_Mpinv) and not(get_P0):
        return M


def acq_to_acq(acqs, param_maps, te=None):
    n_batch,ne,hgt,wdt,n_ch = acqs.shape

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    te_complex = tf.complex(0.0,te) # (nb,ne,1)

    M, M_pinv = gen_M(te) # M shape: (ne,ns)
    M_pinv = tf.squeeze(M_pinv,axis=0)

    # Generate complex signal
    S = tf.complex(acqs[:,:,:,:,0],acqs[:,:,:,:,1]) # (nb,ne,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # shape: (nb,ne,nv)

    r2s = param_maps[:,0,:,:,1] * r2_sc
    phi = param_maps[:,0,:,:,0] * fm_sc
    # r2s = tf.nn.relu(r2s)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1]) # shape: (nb,nv)
    xi_rav = tf.expand_dims(xi_rav,1) # shape: (nb,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (nb,ne,nv)
    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))

    # Matrix operations
    WmS = Wm * Smtx # shape = (nb,ne,nv)
    MWmS = tf.linalg.matmul(M_pinv,WmS) # shape = (nb,ns,nv)
    MMWmS = tf.linalg.matmul(M,MWmS) # shape = (nb,ne,nv)
    Smtx_hat = Wp * MMWmS # shape = (nb,ne,nv)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(MWmS, [n_batch,ns,hgt,wdt,1]) / rho_sc
    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    res_rho = tf.concat([Re_rho,Im_rho],axis=-1)

    # Reshape to original acquisition dimensions
    res_gt = tf.reshape(Smtx_hat, [n_batch,ne,hgt,wdt,1])
    Re_gt = tf.math.real(res_gt)
    Im_gt = tf.math.imag(res_gt)
    res_gt = tf.concat([Re_gt,Im_gt],axis=-1)
    return (res_rho,res_gt)


@tf.custom_gradient
def IDEAL_model(out_maps, params):
    n_batch,_,hgt,wdt,_ = out_maps.shape

    te = params[1] # (nb,ne,1)
    te_complex = tf.complex(0.0,te)
    ne = te.shape[1]

    M = gen_M(te, field=params[0], get_Mpinv=False) # (nb,ne,ns)

    # Generate complex water/fat signals
    real_rho = out_maps[:,:2,:,:,0]
    imag_rho = out_maps[:,:2,:,:,1]
    rho = tf.complex(real_rho, imag_rho) * rho_sc # (nb,ns,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.reshape(rho, [n_batch, ns, num_voxel]) # (nb,ns,nv)

    r2s = tf.nn.relu(out_maps[:,2,:,:,1]) * r2_sc
    phi = out_maps[:,2,:,:,0] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav)) # (nb,ne,nv)

    # Matrix operations
    Mp = tf.linalg.matmul(M, rho_mtx) # (nb,ne,nv)
    Smtx = Wp * Mp # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(Smtx,[n_batch,ne,hgt,wdt])
    S_hat = tf.expand_dims(S_hat, -1)

    # Split into real and imaginary channels
    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    res_gt = tf.concat([Re_gt,Im_gt], axis=-1)

    def grad(upstream, variables=params): # Must be same shape as out_maps
        # Re-format upstream
        upstream = tf.complex(0.5*upstream[:,:,:,:,0],-0.5*upstream[:,:,:,:,1]) # (nb,ne,hgt,wdt)
        upstream = tf.transpose(tf.reshape(upstream, [n_batch,ne,num_voxel]), perm=[0,2,1]) # (nb,nv,ne)

        # Water/fat gradient
        Wp_d = tf.linalg.diag(tf.transpose(Wp,perm=[2,0,1])) # (nv,nb,ne,ne)
        ds_dp = tf.transpose(tf.linalg.matmul(Wp_d,M),perm=[1,0,2,3]) * rho_sc ## (nb,nv,ne,ns) I1

        # Xi gradient, considering Taylor approximation
        dxi = tf.linalg.diag(2*np.pi*tf.squeeze(te_complex,-1)) # (nb,ne,1) --> (nb,ne,ne)
        ds_dxi = tf.linalg.matmul(dxi,Smtx) # (nb,ne,nv)
        ds_dxi = tf.complex(tf.math.real(ds_dxi)*fm_sc,tf.math.imag(ds_dxi)*r2_sc/(2*np.pi))
        ds_dxi = tf.expand_dims(tf.transpose(ds_dxi,perm=[0,2,1]),axis=-1) ## (nb,nv,ne,1) I2

        # Concatenate d_s/d_param gradients
        ds_dq = tf.concat([ds_dp,ds_dxi],axis=-1) # (nb,nv,ne,ns+1)
        ds_dq = tf.transpose(ds_dq, perm=[0,1,3,2]) ## (nb,nv,ns+1,ne)

        grad_res = tf.linalg.matvec(ds_dq, upstream) # (nb,nv,ns+1)
        grad_res = tf.reshape(tf.transpose(grad_res,perm=[0,2,1]), [n_batch,ns+1,hgt,wdt]) # (nb,ns+1,hgt,wdt)
        grad_res_r = +2*tf.math.real(tf.expand_dims(grad_res,axis=-1))
        grad_res_i = -2*tf.math.imag(tf.expand_dims(grad_res,axis=-1))
        grad_res = tf.concat([grad_res_r,grad_res_i],axis=-1) # (nb,ns+1,hgt,wdt,2)

        return (grad_res, [tf.constant([1.0],dtype=tf.float32), tf.ones((n_batch,ne,1),dtype=tf.float32)])

    return res_gt, grad


class IDEAL_Layer(tf.keras.layers.Layer):
    def __init__(self, field=1.5):
        super(IDEAL_Layer, self).__init__()
        self.field = field

    def call(self,out_maps,te=None,ne=6,training=None):
        if te is None:
            te = gen_TEvar(ne, out_maps.shape[0], orig=True)
        return IDEAL_model(out_maps, [self.field, te])


class LWF_Layer(tf.keras.layers.Layer):
    def __init__(self,n_ech,MEBCRN=False):
        super(LWF_Layer, self).__init__()
        self.n_ech = n_ech
        self.MEBCRN = MEBCRN

    def call(self,out_maps,te=None,training=None):
        n_batch,_,hgt,wdt,_ = out_maps.shape
        ne = 6

        if te is None:
            te = gen_TEvar(ne)
        te_complex = tf.complex(0.0,te) # (1,ne)

        M = gen_M(te,get_Mpinv=False) # (nb,ne,ns)

        # Generate complex water/fat signals
        real_rho = tf.transpose(out_maps[:,:,:,:,0],perm=[0,2,3,1])
        imag_rho = tf.transpose(out_maps[:,:,:,:,1],perm=[0,2,3,1])
        rho = tf.complex(real_rho,imag_rho) * rho_sc

        voxel_shape = tf.convert_to_tensor((hgt,wdt))
        num_voxel = tf.math.reduce_prod(voxel_shape)
        rho_mtx = tf.transpose(tf.reshape(rho, [n_batch, num_voxel, ns]), perm=[0,2,1]) # (nb,ns,nv)

        # Matrix operations
        Smtx = tf.linalg.matmul(M,rho_mtx) # (nb,ne,nv)

        # Reshape to original acquisition dimensions
        S_hat = tf.reshape(Smtx,[n_batch,ne,hgt,wdt]) # (nb*ne,hgt,wdt)
        S_hat = tf.expand_dims(S_hat,-1)

        # Split into real and imaginary channels
        Re_gt = tf.math.real(S_hat)
        Im_gt = tf.math.imag(S_hat)
        res_gt = tf.concat([Re_gt,Im_gt],axis=-1)

        return res_gt


@tf.custom_gradient
def IDEAL_mag(out_maps, params):
    n_batch,_,hgt,wdt,_ = out_maps.shape

    te = params[1] # (nb,ne,1)
    te_complex = tf.complex(0.0,te)
    ne = te.shape[1]

    M = gen_M(te, field=params[0], get_Mpinv=False) # (nb,ne,ns)

    # Generate complex water/fat signals
    mag_rho = out_maps[:,0,:,:,:2]
    rho = tf.complex(mag_rho,0.0) * rho_sc # (nb,hgt,wdt,ns)
    rho = tf.transpose(rho,perm=[0,3,1,2]) # (nb,ns,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.reshape(rho, [n_batch, ns, num_voxel]) # (nb,ns,nv)

    pha_rho = tf.complex(0.0,out_maps[:,1,:,:,1]) * 3*np.pi
    pha_rho_rav = tf.reshape(pha_rho, [n_batch, -1]) # (nb,nv)
    pha_rho_rav = tf.expand_dims(pha_rho_rav,1) # (nb,1,nv)
    exp_ph = tf.linalg.matmul(tf.ones([n_batch,ne,1],dtype=tf.complex64), pha_rho_rav) # (nb,ne,nv)

    r2s = out_maps[:,0,:,:,2] * r2_sc
    phi = out_maps[:,1,:,:,2] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav) + exp_ph) # (nb,ne,nv)

    # Matrix operations
    Mp = tf.linalg.matmul(M, rho_mtx) # (nb,ne,nv)
    Smtx = Wp * Mp # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(Smtx,[n_batch,ne,hgt,wdt])
    S_hat = tf.expand_dims(S_hat, -1)

    # Split into real and imaginary channels
    Re_gt = tf.math.real(S_hat)
    Im_gt = tf.math.imag(S_hat)
    res_gt = tf.concat([Re_gt,Im_gt], axis=-1)

    def grad(upstream, variables=params): # Must be same shape as out_maps
        # Re-format upstream
        upstream = tf.complex(0.5*upstream[:,:,:,:,0],-0.5*upstream[:,:,:,:,1]) # (nb,ne,hgt,wdt)
        upstream = tf.transpose(tf.reshape(upstream, [n_batch,ne,num_voxel]), perm=[0,2,1]) # (nb,nv,ne)

        # Water/fat gradient
        Wp_d = tf.linalg.diag(tf.transpose(Wp,perm=[2,0,1])) # (nv,nb,ne,ne)
        ds_dp = tf.transpose(tf.linalg.matmul(Wp_d,M),perm=[1,0,3,2]) * rho_sc ## (nb,nv,ns,ne) I1
        grad_res_rho = +2*tf.math.real(tf.linalg.matvec(ds_dp, upstream)) # (nb,nv,ns)

        # Xi gradient
        dxi = tf.linalg.diag(2*np.pi*tf.squeeze(te_complex,-1)) # (nb,ne,1) --> (nb,ne,ne)
        ds_dxi = tf.linalg.matmul(dxi,Smtx) # (nb,ne,nv)
        ds_dxi = tf.complex(tf.math.real(ds_dxi)*fm_sc,tf.math.imag(ds_dxi)*r2_sc/(2*np.pi))
        ds_dxi = tf.expand_dims(tf.transpose(ds_dxi,perm=[0,2,1]),axis=-2) ## (nb,nv,1,ne) I2
        grad_res_xi = tf.linalg.matvec(ds_dxi, upstream) # (nb,nv,1)
        grad_res_r2 = tf.math.imag(grad_res_xi)
        grad_res_fm = tf.math.real(grad_res_xi)

        # Phi gradient
        dphi = tf.linalg.diag(tf.math.exp(tf.complex(0.0,tf.ones([n_batch,ne],dtype=tf.float32)))) # (nb,ne,ne)
        ds_dphi = tf.linalg.matmul(dphi,Smtx) * 3*np.pi # (nb,ne,nv)
        ds_dphi = tf.expand_dims(tf.transpose(ds_dphi,perm=[0,2,1]),axis=-2) ## (nb,nv,1,ne) I3
        grad_res_phi = tf.math.real(tf.linalg.matvec(ds_dphi, upstream)) # (nb,nv,1)

        # Concatenate d_s/d_param gradients
        grad_res_mag = tf.concat([grad_res_rho,grad_res_r2],axis=-1) # (nb,nv,ns+1)
        grad_res_pha = tf.concat([tf.zeros_like(grad_res_phi),grad_res_phi,grad_res_fm],axis=-1)
        grad_res_mag = tf.expand_dims(tf.reshape(grad_res_mag,[n_batch,hgt,wdt,ns+1]),axis=1) # (nb,1,hgt,wdt,ns+1)
        grad_res_pha = tf.expand_dims(tf.reshape(grad_res_pha,[n_batch,hgt,wdt,ns+1]),axis=1)
        grad_res = tf.concat([grad_res_mag,grad_res_pha],axis=1) # (nb,2,hgt,wdt,ns+1)

        return (grad_res, [tf.constant([1.0],dtype=tf.float32), tf.ones((n_batch,ne,1),dtype=tf.float32)])

    return res_gt, grad


class IDEAL_mag_Layer(tf.keras.layers.Layer):
    def __init__(self, field=1.5):
        super(IDEAL_mag_Layer, self).__init__()
        self.field = field

    def call(self,out_maps,te=None,ne=6,training=None):
        if te is None:
            te = gen_TEvar(ne, out_maps.shape[0], orig=True)
        return IDEAL_mag(out_maps, [self.field, te])


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


def get_rho(acqs, param_maps, field=1.5, te=None, MEBCRN=True):
    if MEBCRN:
        n_batch,ne,hgt,wdt,n_ch = acqs.shape
    else:
        n_batch,hgt,wdt,ech_idx = acqs.shape
        ne = ech_idx//2

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    te_complex = tf.complex(0.0,te) # (nb,ne,1)

    M, M_pinv = gen_M(te, field=field) # M shape: (nb,ne,ns)

    # Generate complex signal
    if MEBCRN:
        S = tf.complex(acqs[:,:,:,:,0],acqs[:,:,:,:,1]) # (nb,ne,hgt,wdt)
    else:
        S = tf.complex(acqs[:,:,:,0::2],acqs[:,:,:,1::2]) # (nb,hgt,wdt,ne)
        S = tf.transpose(S, perm=[0,3,1,2]) # (nb,ne,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # (nb,ne,nv)

    if MEBCRN:
        r2s = param_maps[:,:,:,:,1:] * r2_sc
        phi = param_maps[:,:,:,:,:1] * fm_sc
    else:
        r2s = param_maps[:,:,:,:1] * r2_sc
        phi = param_maps[:,:,:,1:] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (nb,ne,nv)

    # Matrix operations
    WmS = Wm * Smtx # (nb,ne,nv)
    MWmS = tf.linalg.matmul(M_pinv,WmS) # (nb,ns,nv)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(MWmS, [n_batch,ns,hgt,wdt]) / rho_sc

    if MEBCRN:
        rho_hat = tf.expand_dims(rho_hat, -1)
        Re_rho = tf.math.real(rho_hat)
        Im_rho = tf.math.imag(rho_hat)
        res_rho = tf.concat([Re_rho,Im_rho],axis=-1)
    else:
        rho_hat = tf.transpose(rho_hat, perm=[0,2,3,1])
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
def PDFF_uncertainty(acqs, mean_maps, var_maps, te=None, MEBCRN=True):
    if MEBCRN:
        n_batch,ne,hgt,wdt,_ = acqs.shape
    else:
        n_batch,hgt,wdt,d_ech = acqs.shape
        ne = d_ech//2

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    M, M_pinv = gen_M(te)

    # te_complex = tf.expand_dims(tf.complex(0.0,te),-1)
    te_real = tf.complex(te,0.0)

    # Generate complex signal
    if MEBCRN:
        real_S = acqs[...,:1]
        imag_S = acqs[...,1:]
    else:
        real_S = acqs[...,0::2]
        imag_S = acqs[...,1::2]
    
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    if MEBCRN:
        Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # (nb,ne,nv)
    else:
        Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1]) # (nb,ne,nv)

    r2s = mean_maps[...,1] * r2_sc
    phi = mean_maps[...,0] * fm_sc
    r2s_unc = var_maps[...,1] * (r2_sc**2)
    phi_unc = var_maps[...,0] * (fm_sc**2)

    r2s_rav = tf.reshape(tf.complex(r2s,0.0),[n_batch,-1])
    r2s_rav = tf.expand_dims(r2s_rav,1) # (nb,1,nv)
    r2s_unc_rav = tf.reshape(tf.complex(r2s_unc,0.0),[n_batch,-1])
    r2s_unc_rav = tf.expand_dims(r2s_unc_rav,1) # (nb,1,nv)
    phi_unc_rav = tf.reshape(tf.complex(phi_unc,0.0),[n_batch,-1])
    phi_unc_rav = tf.expand_dims(phi_unc_rav,1) # (nb,1,nv)

    # Diagonal matrix with the exponential of fieldmap variance
    r2s_var_aux = tf.linalg.matmul(te_real**2, r2s_unc_rav) # (nb,ne,nv)
    Wm_unc_r2s = tf.math.exp(tf.linalg.matmul(2*te_real, r2s_rav) + r2s_var_aux) # (nb,ne,nv)
    Wm_var_r2s = tf.math.exp(r2s_var_aux)
    Wm_var_phi = tf.math.exp(tf.linalg.matmul(-(2*np.pi * te_real)**2, phi_unc_rav)) # (nb,ne,nv)
    Wm_var = -(1 - Wm_var_phi) * (1 - Wm_var_r2s) * Wm_unc_r2s

    # Matrix operations (variance)
    WmZS = Wm_var * (Smtx * tf.math.conj(Smtx))
    MWmZS = tf.linalg.matmul(M_pinv * tf.math.conj(M_pinv),WmZS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    if MEBCRN:
        rho_var = tf.reshape(MWmZS,[n_batch,ns,hgt,wdt,1]) / rho_sc
        Re_rho_var = tf.math.real(rho_var)
        Im_rho_var = tf.math.imag(rho_var)
        res_rho_var = tf.concat([Re_rho_var,Im_rho_var],axis=-1)
    else:
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


#@tf.function
def acq_uncertainty(acqs, mean_maps, var_maps, te=None, MEBCRN=True, rem_R2=False):
    if MEBCRN:
        n_batch,ne,hgt,wdt,_ = acqs.shape
    else:
        n_batch,hgt,wdt,d_ech = acqs.shape
        ne = d_ech//2

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    M, M_pinv = gen_M(te)
    MM = tf.linalg.matmul(M,M_pinv)

    # te_complex = tf.expand_dims(tf.complex(0.0,te),-1)
    te_real = tf.complex(te,0.0)

    # Generate complex signal
    if MEBCRN:
        real_S = acqs[...,:1]
        imag_S = acqs[...,1:]
    else:
        real_S = acqs[...,0::2]
        imag_S = acqs[...,1::2]
    
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    if MEBCRN:
        Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # (nb,ne,nv)
    else:
        Smtx = tf.transpose(tf.reshape(S, [n_batch, num_voxel, ne]), perm=[0,2,1]) # (nb,ne,nv)

    r2s = mean_maps[...,1] * r2_sc
    phi = mean_maps[...,0] * fm_sc
    r2s_unc = var_maps[...,1] * (r2_sc**2)
    phi_unc = var_maps[...,0] * (fm_sc**2)

    r2s_rav = tf.reshape(tf.complex(r2s,0.0),[n_batch,-1])
    r2s_rav = tf.expand_dims(r2s_rav,1) # (nb,1,nv)
    r2s_unc_rav = tf.reshape(tf.complex(r2s_unc,0.0),[n_batch,-1])
    r2s_unc_rav = tf.expand_dims(r2s_unc_rav,1) # (nb,1,nv)
    phi_unc_rav = tf.reshape(tf.complex(phi_unc,0.0),[n_batch,-1])
    phi_unc_rav = tf.expand_dims(phi_unc_rav,1) # (nb,1,nv)

    # Diagonal matrix with the exponential of fieldmap variance
    Wm_var_phi = tf.math.exp(tf.linalg.matmul(-(2*np.pi * te_real)**2, phi_unc_rav)) # (nb,ne,nv)
    Wm_var = -(1 - Wm_var_phi)
    if not(rem_R2):
        r2s_var_aux = tf.linalg.matmul(te_real**2, r2s_unc_rav) # (nb,ne,nv)
        Wm_unc_r2s = tf.math.exp(tf.linalg.matmul(2*te_real, r2s_rav) + r2s_var_aux) # (nb,ne,nv)
        Wm_var_r2s = tf.math.exp(r2s_var_aux)
        Wm_var *= (1 - Wm_var_r2s) * Wm_unc_r2s

    # Matrix operations (variance)
    WmZS = Wm_var * (Smtx * tf.math.conj(Smtx))
    WpMMWmZS = Wm_var * tf.linalg.matmul(MM * tf.math.conj(MM), WmZS)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    if MEBCRN:
        S_var = tf.reshape(WpMMWmZS, [n_batch,ne,hgt,wdt,1])
        # res_S_var = tf.concat([tf.math.real(S_var), tf.math.imag(S_var)],axis=-1)
        res_S_var = tf.concat([tf.abs(S_var), tf.abs(S_var)],axis=-1)
    else:
        S_var = tf.reshape(tf.transpose(WpMMWmZS, perm=[0,2,1]),[n_batch,hgt,wdt,ne])
        Re_S_var = tf.math.real(S_var)
        Im_S_var = tf.math.imag(S_var)
        zero_fill = tf.zeros_like(Re_S_var)
        re_stack_var = tf.stack([Re_S_var,zero_fill],4)
        re_aux_var = tf.reshape(re_stack_var,[n_batch,hgt,wdt,2*ne])
        im_stack_var = tf.stack([zero_fill,Im_S_var],4)
        im_aux_var = tf.reshape(im_stack_var,[n_batch,hgt,wdt,2*ne])
        res_S_var = re_aux_var + im_aux_var

    return res_S_var
