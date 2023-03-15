import numpy as np
import tensorflow as tf
import h5py
h5py._errors.unsilence_errors()

class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


@tf.function
def load_hdf5(hdf5_file,ech_idx=12,start=0,end=2000,
            acqs_data=True,te_data=False,complex_data=False,remove_zeros=True):
    f = h5py.File(hdf5_file.numpy(), 'r')
    if acqs_data:
        acqs = f['Acquisitions'][start:end]
    out_maps = f['OutMaps'][start:end]
    if te_data:
        TEs = f['TEs'][start:end]
    f.close()

    if remove_zeros:
        idxs_list = []
        for nd in range(len(out_maps)):
            if np.sum(out_maps[nd,:,:,0])!=0.0:
                idxs_list.append(nd)
    else:
        idxs_list = [i for i in range(len(out_maps))]

    out_maps = out_maps[idxs_list,:,:,:]

    if acqs_data:
        if complex_data:
            acqs_real = acqs[:,:,:,0::2]
            acqs_imag = acqs[:,:,:,1::2]
            acqs = acqs_real + 1j*acqs_imag
        acqs = acqs[idxs_list,:,:,:ech_idx]
        if te_data:
            if complex_data:
                TEs = TEs[idxs_list,:ech_idx]
            else:
                TEs = TEs[idxs_list,:ech_idx//2]
            return acqs, out_maps, TEs
        else:
            return acqs, out_maps
    elif te_data:
        if complex_data:
            TEs = TEs[idxs_list,:ech_idx]
        else:
            TEs = TEs[idxs_list,:ech_idx//2]
        return out_maps, TEs
    else:
        return out_maps


def gen_hdf5(filepaths,ech_idx,lims_list,acqs_data=True,te_data=False,remove_zeros=True):
    for k in range(len(filepaths)):
        file = filepaths[k]
        lims = lims_list[k]
        with h5py.File(file, 'r') as f:
            if lims[1] >= lims[0]:
                idx_list = np.arange(lims[0],lims[1])
            else:
                idx_list = np.concatenate([np.arange(0,lims[1]),np.arange(lims[0],len(f['OutMaps']))])
            for i in idx_list:
                out = f['OutMaps'][i]
                if tf.reduce_sum(out)!=0.0 or not(remove_zeros):
                    if acqs_data:
                        im = f['Acquisitions'][i,:,:,:ech_idx]
                        yield im, out
                    elif te_data:
                        TEs = f['TEs'][i]
                        yield im, out, TEs
                    else:
                        yield out
            f.close()
        

def group_TEs(A,B,TEs,TE1,dTE):
    TE1 = np.float32(TE1)
    dTE = np.float32(dTE)
    TE1_orig = np.float32(0.0013)
    dTE_orig = np.float32(0.0021)

    len_dataset,hgt,wdt,d_ech = A.shape
    _,_,_,n_out = B.shape

    num_pat = 0
    all_null_slices = [] # To save the indexes of the set-to-zero slices
    orig_slices = []
    all_sel_slices = [] # To save the indexes of the corresponding slices per patient
    sel_slices = []

    flag_orig = False
    flag_sel = False
    flag_noTE = True

    for idx in range(len_dataset+1):
        if idx < len_dataset:
            TE1_i = np.round(TEs[idx,0],4)
            dTE_i = np.round(np.mean(np.diff(TEs[idx,:])),4)
        else:
            TE1_i = TE1_orig
            dTE_i = dTE_orig

        # Check if it corresponds to original TEs
        if TE1_i==TE1_orig and dTE_i==dTE_orig:
            if not(flag_orig):
                flag_orig = True
                if num_pat > 0:
                    # Si paciente anterior no tenía imags en comb TEs seleccionada:
                    # Guardar índices de slice orig (paciente anterior) y convertir a 0
                    if flag_noTE:
                        for os in orig_slices:
                            all_null_slices.append(os)
                            all_sel_slices.append(os)
                    else:
                        flag_noTE = True
                        for ss in sel_slices:
                            all_sel_slices.append(ss)
                        sel_slices = []
                num_pat += 1
                orig_slices = []
            orig_slices.append(idx)
        else:
            if flag_orig:
                flag_orig = False

        # Check if it corresponds to selected TEs
        if TE1_i==TE1 and dTE_i==dTE:
            if not(flag_sel):
                flag_sel = True
                flag_noTE = False
            sel_slices.append(idx)
        else:
            if flag_sel:
                flag_sel = False

    A[all_null_slices,:,:,:] = 0.0
    B[all_null_slices,:,:,:] = 0.0

    A = A[all_sel_slices,:,:,:]
    B = B[all_sel_slices,:,:,:]
    TEs = TEs[all_sel_slices,:]

    return A, B, TEs

