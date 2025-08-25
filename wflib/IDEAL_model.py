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
def gen_M(te, field=1.5, get_Mpinv=True, get_P0=False, get_H=False):
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
        if get_H:
            H = tf.math.real(tf.linalg.matmul(M_pinv,M))
            Q_h, R_h = tf.linalg.qr(H)
            H_pinv = tf.linalg.solve(R_h, tf.transpose(Q_h, perm=[0,2,1]))
            H_pinv = tf.cast(H_pinv,tf.complex64)

    if get_P0 and get_Mpinv:
        return M, P0, M_pinv
    elif get_Mpinv and not(get_P0) and not(get_H):
        return M, M_pinv
    elif get_Mpinv and not(get_P0):
        return M, M_pinv, H_pinv
    elif not(get_Mpinv) and not(get_P0) and not(get_H):
        return M


def acq_to_acq(acqs, param_maps, te=None, only_mag=False):
    n_batch,ne,hgt,wdt,n_ch = acqs.shape

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    te_complex = tf.complex(0.0,te) # (nb,ne,1)

    M, M_pinv = gen_M(te) # M shape: (nb,ne,ns)

    # Generate complex signal
    S = tf.complex(acqs[:,:,:,:,0],acqs[:,:,:,:,1]) # (nb,ne,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # shape: (nb,ne,nv)

    r2s = param_maps[:,0,:,:,1] * r2_sc
    phi = param_maps[:,0,:,:,0] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1]) # shape: (nb,nv)
    xi_rav = tf.expand_dims(xi_rav,1) # shape: (nb,1,nv)
    if only_mag:
        r2s_rav = tf.reshape(r2s,[n_batch,-1]) # shape: (nb,nv)
        r2s_rav = tf.expand_dims(r2s_rav,1) # shape: (nb,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (nb,ne,nv)
    if only_mag:
        Wp = tf.math.exp(tf.linalg.matmul(-te, r2s_rav))
    else:
        Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav))

    # Matrix operations
    WmS = Wm * Smtx # shape = (nb,ne,nv)
    MWmS = tf.linalg.matmul(M_pinv,WmS) # shape = (nb,ns,nv)
    MMWmS = tf.linalg.matmul(M,MWmS) # shape = (nb,ne,nv)
    if only_mag:
        MMWmS = tf.abs(MMWmS) # shape = (nb,ne,nv)
    Smtx_hat = Wp * MMWmS # shape = (nb,ne,nv)

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    rho_hat = tf.reshape(MWmS, [n_batch,ns,hgt,wdt,1]) / rho_sc
    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    res_rho = tf.concat([Re_rho,Im_rho],axis=-1)

    # Reshape to original acquisition dimensions
    res_gt = tf.reshape(Smtx_hat, [n_batch,ne,hgt,wdt,1])
    if not(only_mag):
        Re_gt = tf.math.real(res_gt)
        Im_gt = tf.math.imag(res_gt)
        res_gt = tf.concat([Re_gt,Im_gt],axis=-1)
    return (res_rho,res_gt)


#@tf.custom_gradient
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

    if out_maps.shape[1] > 3:
        pha_bip = tf.complex(out_maps[:,-1,:,:,0],0.0) * np.pi
        pha_bip_rav = tf.reshape(pha_bip, [n_batch, -1])
        pha_bip_rav = tf.expand_dims(pha_bip_rav,1)
        pha_tog = tf.range(1,ne+1,dtype=tf.float32)
        bip_cnst = tf.pow(-tf.ones([n_batch,ne,1],dtype=tf.float32),tf.expand_dims(pha_tog,axis=-1))
        bip_cnst = tf.complex(0.0,bip_cnst)
        exp_ph = tf.linalg.matmul(bip_cnst, pha_bip_rav) # (nb,ne,nv)
    else:
        exp_ph = tf.constant(0.0,dtype=tf.complex64)

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

    # def grad(upstream, variables=params): # Must be same shape as out_maps
    #     # Re-format upstream
    #     upstream = tf.complex(0.5*upstream[:,:,:,:,0],-0.5*upstream[:,:,:,:,1]) # (nb,ne,hgt,wdt)
    #     upstream = tf.transpose(tf.reshape(upstream, [n_batch,ne,num_voxel]), perm=[0,2,1]) # (nb,nv,ne)

    #     # Water/fat gradient
    #     Wp_d = tf.linalg.diag(tf.transpose(Wp,perm=[2,0,1])) # (nv,nb,ne,ne)
    #     ds_dp = tf.transpose(tf.linalg.matmul(Wp_d,M),perm=[1,0,2,3]) * rho_sc ## (nb,nv,ne,ns) I1

    #     # Xi gradient, considering Taylor approximation
    #     dxi = tf.linalg.diag(2*np.pi*tf.squeeze(te_complex,-1)) # (nb,ne,1) --> (nb,ne,ne)
    #     ds_dxi = tf.linalg.matmul(dxi,Smtx) # (nb,ne,nv)
    #     ds_dxi = tf.complex(tf.math.real(ds_dxi)*fm_sc,tf.math.imag(ds_dxi)*r2_sc/(2*np.pi))
    #     ds_dxi = tf.expand_dims(tf.transpose(ds_dxi,perm=[0,2,1]),axis=-1) ## (nb,nv,ne,1) I2

    #     # Concatenate d_s/d_param gradients
    #     ds_dq = tf.concat([ds_dp,ds_dxi],axis=-1) # (nb,nv,ne,ns+1)
    #     ds_dq = tf.transpose(ds_dq, perm=[0,1,3,2]) ## (nb,nv,ns+1,ne)

    #     grad_res = tf.linalg.matvec(ds_dq, upstream) # (nb,nv,ns+1)
    #     grad_res = tf.reshape(tf.transpose(grad_res,perm=[0,2,1]), [n_batch,ns+1,hgt,wdt]) # (nb,ns+1,hgt,wdt)
    #     grad_res_r = +2*tf.math.real(tf.expand_dims(grad_res,axis=-1))
    #     grad_res_i = -2*tf.math.imag(tf.expand_dims(grad_res,axis=-1))
    #     grad_res = tf.concat([grad_res_r,grad_res_i],axis=-1) # (nb,ns+1,hgt,wdt,2)

    #     return (grad_res, [tf.constant([1.0],dtype=tf.float32), tf.ones((n_batch,ne,1),dtype=tf.float32)])

    return res_gt #, grad


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


def IDEAL_mag(out_maps, params):
    n_batch,_,hgt,wdt,_ = out_maps.shape
    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)

    te = params[1] # (nb,ne,1)
    te_complex = tf.complex(0.0,te)
    ne = te.shape[1]

    M = gen_M(te, field=params[0], get_Mpinv=False) # (nb,ne,ns)
    
    # Extract PDFF, PD, and R2s
    ff = out_maps[:,0,:,:,:1] # (nb,hgt,wdt,1)
    pd = out_maps[:,1,:,:,:1] # (nb,hgt,wdt,1)
    r2s = out_maps[:,1,:,:,1] * r2_sc # (nb,hgt,wdt)

    # Extract common WF phase and off-res
    pha_rho = tf.complex(0.0,out_maps[:,2,:,:,:1]) * np.pi * 4 # (nb,hgt,wdt,1)
    phi = out_maps[:,2,:,:,1] * fm_sc

    rho_w = tf.complex((1.0 - ff) * pd * rho_sc, 0.0)
    rho_w *= tf.math.exp(pha_rho)
    rho_f = tf.complex(ff * pd * rho_sc, 0.0)
    rho_f *= tf.math.exp(pha_rho)
    rho = tf.concat([rho_w,rho_f],axis=-1) # (nb,hgt,wdt,ns)
    rho = tf.transpose(rho,perm=[0,3,1,2]) # (nb,ns,hgt,wdt)

    rho_mtx = tf.reshape(rho, [n_batch, ns, num_voxel]) # (nb,ns,nv)

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

    return res_gt


def IDEAL_mag_phase(out_maps, params):
    n_batch,_,hgt,wdt,_ = out_maps.shape
    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    
    te = params[1] # (nb,ne,1)
    te_complex = tf.complex(0.0,te)
    ne = te.shape[1]

    M = gen_M(te, field=params[0], get_Mpinv=False) # (nb,ne,ns)
    
    # Generate complex water/fat signals
    mag_rho = out_maps[:,0,:,:,:2]
    rho = tf.complex(mag_rho,0.0) * rho_sc # (nb,hgt,wdt,ns)
    rho = tf.transpose(rho,perm=[0,3,1,2]) # (nb,ns,hgt,wdt)

    pha_rho = tf.complex(0.0,out_maps[:,1,:,:,:2]) * 4 * np.pi
    pha_rho = tf.transpose(pha_rho,perm=[0,3,1,2]) # (nb,ns,hgt,wdt)
    
    rho *= tf.math.exp(pha_rho)
    rho_mtx = tf.reshape(rho, [n_batch, ns, num_voxel]) # (nb,ns,nv)

    r2s = out_maps[:,0,:,:,2] * r2_sc
    phi = out_maps[:,1,:,:,2] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    pha_bip = tf.complex(out_maps[:,1,:,:,3:],0.0) * 4 * np.pi
    pha_bip_rav = tf.reshape(pha_bip, [n_batch, -1])
    pha_bip_rav = tf.expand_dims(pha_bip_rav,1)
    pha_tog = tf.range(1,ne+1,dtype=tf.float32)
    bip_cnst = tf.pow(-tf.ones([n_batch,ne,1],dtype=tf.float32),tf.expand_dims(pha_tog,axis=-1))
    bip_cnst = tf.complex(0.0,bip_cnst)
    exp_ph = tf.linalg.matmul(bip_cnst, pha_bip_rav) # (nb,ne,nv)

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

    return res_gt


class IDEAL_mag_Layer(tf.keras.layers.Layer):
    def __init__(self, field=1.5, sep_phase=False,):
        super(IDEAL_mag_Layer, self).__init__()
        self.field = field
        self.sep_phase = sep_phase

    def call(self,out_maps,te=None,ne=6,training=None):
        if te is None:
            te = gen_TEvar(ne, out_maps.shape[0], orig=True)
        if self.sep_phase:
            return IDEAL_mag_phase(out_maps, [self.field, te])
        else:
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


def get_rho(acqs, param_maps, field=1.5, te=None, phase_constraint=False, MEBCRN=True, acq_demod=False):
    if MEBCRN:
        n_batch,ne,hgt,wdt,n_ch = acqs.shape
    else:
        n_batch,hgt,wdt,ech_idx = acqs.shape
        ne = ech_idx//2

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    te_complex = tf.complex(0.0,te) # (nb,ne,1)

    if phase_constraint:
        M, M_pinv, H = gen_M(te, field=field, get_H=True) # M shape: (nb,ne,ns)
    else:
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
        r2s = param_maps[:,:1,:,:,1:] * r2_sc
        phi = param_maps[:,:1,:,:,:1] * fm_sc
    else:
        r2s = param_maps[:,:,:,:1] * r2_sc
        phi = param_maps[:,:,:,1:] * fm_sc

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi,r2s/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    if param_maps.shape[1] > 3:
        pha_bip = tf.complex(param_maps[:,-1,:,:,0],0.0) * np.pi
        pha_bip_rav = tf.reshape(pha_bip, [n_batch, -1])
        pha_bip_rav = tf.expand_dims(pha_bip_rav,1)
        pha_tog = tf.range(1,ne+1,dtype=tf.float32)
        bip_cnst = tf.pow(-tf.ones([n_batch,ne,1],dtype=tf.float32),tf.expand_dims(pha_tog,axis=-1))
        bip_cnst = tf.complex(0.0,bip_cnst)
        exp_ph = tf.linalg.matmul(bip_cnst, pha_bip_rav) # (nb,ne,nv)
    else:
        exp_ph = tf.constant(0.0,dtype=tf.complex64)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav) - exp_ph) # shape = (nb,ne,nv)

    # Matrix operations
    WmS = Wm * Smtx # (nb,ne,nv)

    # Extract corresponding Water/Fat signals
    if phase_constraint:
        MWmS_L = tf.linalg.matmul(M_pinv,WmS) # (nb,ns,nv)
        HMWmS = tf.linalg.matmul(H,MWmS_L) # (nb,ns,nv)
        MHMWmS = tf.reduce_sum(MWmS_L * HMWmS,axis=1,keepdims=True) # (nb,1,nv)
        rho_pha = 0.5*tf.math.angle(MHMWmS) # (nb,1,nv)
        rho_pha = tf.concat([rho_pha,rho_pha],axis=-2) # (nb,ns,nv) - Replicate for water and fat
        real_MWmS = tf.math.real(MWmS_L*tf.math.exp(tf.complex(0.0,-rho_pha))) # (nb,ns,nv)
        rho_mag = tf.linalg.matmul(tf.abs(H),real_MWmS) # (nb,ns,nv)
        MWmS = tf.complex(rho_mag,0.0)*tf.math.exp(tf.complex(0.0,rho_pha))
    else:
        MWmS = tf.linalg.matmul(M_pinv,WmS) # (nb,ns,nv)
    
    rho_hat = tf.reshape(MWmS, [n_batch,ns,hgt,wdt]) / rho_sc

    
    # Reshape to original images dimensions
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

    if acq_demod:
        # Reshape to original acquisition dimensions
        res_gt = tf.reshape(WmS, [n_batch,ne,hgt,wdt,1])
        Re_gt = tf.math.real(res_gt)
        Im_gt = tf.math.imag(res_gt)
        res_gt = tf.concat([Re_gt,Im_gt],axis=-1)
        return (res_rho,res_gt)
    else:
        return res_rho


#@tf.function
def PDFF_uncertainty(acqs, phi_tfp, r2s_tfp, te=None, rem_R2=False):
    n_batch,ne,hgt,wdt,_ = acqs.shape
    
    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    M, P0, M_pinv = gen_M(te, get_P0=True)
    MTM =  tf.linalg.matmul(tf.transpose(M, perm=[0,2,1], conjugate=True),M) # (nb,ns,ns)
    # MTM_wf = tf.expand_dims(tf.linalg.diag_part(MTM_inv),axis=-1)

    te_complex = tf.complex(0.0,te)
    te_real = tf.complex(te,0.0)

    # Generate complex signal
    real_S = acqs[...,:1]
    imag_S = acqs[...,1:]
    
    S = tf.complex(real_S,imag_S)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(S, [n_batch, ne, num_voxel]) # (nb,ne,nv)

    phi_mean = phi_tfp.mean() * fm_sc
    phi_var = phi_tfp.variance() * (fm_sc**2)

    if rem_R2:
        r2s_mean = tf.zeros_like(phi_mean)
        r2s_var = tf.zeros_like(phi_var)
    else:
        r2s_mean = r2s_tfp.mean() * r2_sc
        r2s_var = r2s_tfp.variance() * (r2_sc**2)

    # IDEAL Operator evaluation for xi = phi + 1j*r2s/(2*np.pi)
    xi = tf.complex(phi_mean,r2s_mean/(2*np.pi))
    xi_rav = tf.reshape(xi,[n_batch,-1])
    xi_rav = tf.expand_dims(xi_rav,1) # (nb,1,nv)

    Wm = tf.math.exp(tf.linalg.matmul(-2*np.pi * te_complex, xi_rav)) # shape = (nb,ne,nv)
    Wp = tf.math.exp(tf.linalg.matmul(+2*np.pi * te_complex, xi_rav)) # shape = (nb,ne,nv)

    r2s_mu_rav = tf.expand_dims(tf.reshape(r2s_mean,[n_batch,-1]),1) # (nb,1,nv)
    r2s_sigma_rav = tf.expand_dims(tf.reshape(r2s_var,[n_batch,-1]),1) # (nb,1,nv)
    phi_sigma_rav = tf.expand_dims(tf.reshape(phi_var,[n_batch,-1]),1) # (nb,1,nv)

    # Diagonal matrix with the exponential of fieldmap variance
    Wm_var = 1 - tf.math.exp(tf.linalg.matmul(-(2*np.pi * te)**2, phi_sigma_rav)) # (nb,ne,nv) NEG
    if not(rem_R2):
        r2s_var_aux = tf.math.exp(tf.linalg.matmul(te, r2s_mu_rav))
        r2s_var_aux *= tf.linalg.matmul(te**2, r2s_sigma_rav)
        Wm_var += r2s_var_aux # (nb,ne,nv) NEG

    # Matrix operations (variance)
    WpP0Wm = Wp * tf.linalg.matmul(P0, Wm) # (nb,ne,nv)
    s_var = tf.abs(tf.math.conj(WpP0Wm) * WpP0Wm) # (nb,ne,nv)
    y_Sigma = Wm_var * s_var 
    y_Sigma += Wm_var * tf.abs(tf.math.conj(Smtx) * Smtx)
    # y_Sigma += tf.abs(tf.math.conj(Wm) * Wm) * s_var
    y_Sigma_inv = tf.math.divide_no_nan(tf.ones_like(y_Sigma),y_Sigma) # (nb,ne,nv)
    y_Sigma_inv = tf.linalg.diag(tf.transpose(y_Sigma_inv,perm=[2,0,1])) # (nv,nb,ne,ne)
    SigM = tf.linalg.matmul(tf.complex(y_Sigma_inv,0.0),M) # (nv,nb,ne,ns)
    MTSigM = tf.linalg.matmul(tf.transpose(M, perm=[0,2,1], conjugate=True),SigM) # (nv,nb,ns,ns)
    rho_cov = tf.linalg.inv(MTSigM)
    # rho_var = tf.transpose(tf.linalg.diag_part(rho_cov),perm=[1,2,0]) # (nb,ns,nv)
    rho_var = tf.transpose(tf.reshape(rho_cov,[num_voxel,n_batch,ns*ns]),perm=[1,2,0]) # (nb,ns^2,nv)

    y_samp = tf.transpose(Wm * Smtx, perm=[2,0,1]) # shape = (nv,nb,ne)
    SigY = tf.linalg.matmul(tf.complex(y_Sigma_inv,0.0),tf.expand_dims(y_samp,axis=-1)) # shape = (nv,nb,ne,1)
    MTSigY = tf.linalg.matmul(tf.transpose(M, perm=[0,2,1], conjugate=True),SigY) # shape = (nv,nb,ns,1)
    rho_hat = tf.linalg.matmul(rho_cov,MTSigY) # shape = (nv,nb,ns,1)
    rho_hat = tf.reshape(tf.transpose(rho_hat,perm=[1,2,0,3]), [n_batch,ns,hgt,wdt,1]) / rho_sc

    # Extract corresponding Water/Fat signals
    # Reshape to original images dimensions
    Re_rho = tf.math.real(rho_hat)
    Im_rho = tf.math.imag(rho_hat)
    res_rho = tf.concat([Re_rho,Im_rho],axis=-1)
    res_rho_var = tf.reshape(tf.abs(rho_var),[n_batch,ns*ns,hgt,wdt,1]) / (rho_sc**2)
    return res_rho, res_rho_var


#@tf.function
def acq_uncertainty(rho_maps, phi_tfp, r2s_tfp, ne=6, te=None, rem_R2=False, only_mag=False):
    n_batch,_,hgt,wdt,n_ch = rho_maps.shape

    if te is None:
        te = gen_TEvar(ne+1, bs=n_batch, orig=True) # (nb,ne,1)
        te = te[:,1:,:]

    M = gen_M(te, get_Mpinv=False)

    # Generate complex water/fat signals
    real_rho = rho_maps[:,:2,:,:,0]
    imag_rho = rho_maps[:,:2,:,:,1]
    rho = tf.complex(real_rho, imag_rho) * rho_sc # (nb,ns,hgt,wdt)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    rho_mtx = tf.reshape(rho, [n_batch, ns, num_voxel]) # (nb,ns,nv)
    
    # phi_mean = phi_tfp.mean() * fm_sc
    phi_var = phi_tfp.variance() * (fm_sc**2)

    if rem_R2:
        r2s_mean = tf.zeros_like(phi_var)
        r2s_var = tf.zeros_like(phi_var)
    else:
        r2s_mean = r2s_tfp.mean() * r2_sc
        r2s_var = r2s_tfp.variance() * (r2_sc**2)
        if r2s_mean.shape[-1] > 1:
            r2s_mean = r2s_mean[...,:1]
            r2s_var = r2s_var[...,:1] 

    r2s_mu_rav = tf.expand_dims(tf.reshape(r2s_mean,[n_batch,-1]),1) # (nb,1,nv)
    r2s_sigma_rav = tf.expand_dims(tf.reshape(r2s_var,[n_batch,-1]),1) # (nb,1,nv)
    phi_sigma_rav = tf.expand_dims(tf.reshape(phi_var,[n_batch,-1]),1) # (nb,1,nv)

    # Diagonal matrix with the exponential of fieldmap variance
    Wp_var = 1 - tf.math.exp(tf.linalg.matmul(-(2*np.pi * te)**2, phi_sigma_rav)) # (nb,ne,nv) NEG
    if not(rem_R2):
        r2s_var_aux = tf.math.exp(tf.linalg.matmul(-te, r2s_mu_rav))
        r2s_var_aux = tf.linalg.matmul(te**2, r2s_sigma_rav) #*
        Wp_var += r2s_var_aux # (nb,ne,nv) NEG

    # Matrix operations (variance)
    MMWmS = tf.linalg.matmul(M,rho_mtx) # shape = (nb,ne,nv)
    Smtx_var = Wp_var * tf.abs(MMWmS * tf.math.conj(MMWmS)) # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    res_S_var = tf.expand_dims(tf.reshape(Smtx_var,[n_batch,ne,hgt,wdt]), -1)
    if not(only_mag):
        res_S_var = tf.concat([res_S_var,res_S_var], axis=-1) # Same variance for real/imag

    return res_S_var


def acq_mag_demod(acqs_abs, out_maps, te=None):
    n_batch,_,hgt,wdt,_ = out_maps.shape
    ne = acqs_abs.shape[1]

    if te is None:
        te = gen_TEvar(ne, bs=n_batch, orig=True) # (nb,ne,1)

    M = gen_M(te, get_Mpinv=False) # (nb,ne,ns)

    # Generate complex water/fat signals
    abs_rho = out_maps[:,:2,:,:,0] # (nb,ns,hgt,wdt)
    sum_rho = tf.reduce_sum(abs_rho, axis=1, keepdims=True)
    sum_rho = tf.concat([sum_rho,sum_rho],axis=1)
    abs_rho = tf.math.divide_no_nan(abs_rho,sum_rho)
    abs_rho_complex = tf.complex(abs_rho,0.0)

    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)
    Smtx = tf.reshape(tf.squeeze(acqs_abs,axis=-1), [n_batch, ne, num_voxel]) # (nb,ne,nv)
    rho_mtx = tf.reshape(abs_rho_complex, [n_batch, ns, num_voxel]) # (nb,ns,nv)

    # Matrix operations
    Mp = tf.abs(tf.linalg.matmul(M, rho_mtx)) # (nb,ne,nv)
    Smtx_demod = tf.math.divide_no_nan(Smtx,Mp) # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(Smtx_demod,[n_batch,ne,hgt,wdt])
    res_gt = tf.expand_dims(S_hat, -1)

    return res_gt 


def recon_demod_abs(ech1_abs, out_maps, te=None):
    n_batch,_,hgt,wdt,_ = out_maps.shape
    voxel_shape = tf.convert_to_tensor((hgt,wdt))
    num_voxel = tf.math.reduce_prod(voxel_shape)

    if te is None:
        te = gen_TEvar(6, bs=n_batch, orig=True) # (nb,ne,1)
        te = te[:,1:,:] - te[0,0,0]
    ne = te.shape[1]

    # Generate complex water/fat signals
    abs_rho = ech1_abs[:,0,:,:,0] # (nb,hgt,wdt)
    rho_rav = tf.expand_dims(tf.reshape(abs_rho,[n_batch,-1]),1) # (nb,1,nv)
    rho_mtx = tf.repeat(rho_rav,ne,axis=1) # (nb,ne,nv)
    
    r2s_tfp = out_maps[:,0,:,:,0] * r2_sc # (nb,hgt,wdt)
    r2s_mu_rav = tf.expand_dims(tf.reshape(r2s_tfp,[n_batch,-1]),1) # (nb,1,nv)
    
    W = tf.math.exp(tf.linalg.matmul(te, -r2s_mu_rav)) # (nb,ne,nv)
    Smtx = rho_mtx * W # (nb,ne,nv)

    # Reshape to original acquisition dimensions
    S_hat = tf.reshape(Smtx,[n_batch,ne,hgt,wdt])
    res_gt = tf.expand_dims(S_hat, -1)

    return res_gt