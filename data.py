import numpy as np
import tensorflow as tf
import h5py

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

def load_hdf5(ds_dir,hdf5_file,ech_idx,acqs_data=True,te_data=False,complex_data=False):
    f = h5py.File(ds_dir + hdf5_file, 'r')
    if acqs_data:
        acqs = f['Acquisitions'][...]
    out_maps = f['OutMaps'][...]
    if te_data:
        TEs = f['TEs'][...]
    f.close()

    idxs_list = []
    for nd in range(len(out_maps)):
        if np.sum(out_maps[nd,:,:,1])!=0.0:
            idxs_list.append(nd)

    out_maps = out_maps[idxs_list,:,:,:]

    if acqs_data and not(te_data):
        if complex_data:
            acqs_real = acqs[:,:,:,0::2]
            acqs_imag = acqs[:,:,:,1::2]
            acqs = acqs_real + 1j*acqs_imag
        acqs = acqs[idxs_list,:,:,:ech_idx]
        return acqs, out_maps
    elif te_data:
        if complex_data:
            TEs = TEs[idxs_list,:ech_idx]
        else:
            TEs = TEs[idxs_list,:ech_idx//2]
        if acqs_data:
            return acqs, out_maps, TEs
        else:
            return out_maps, TEs
    elif not(acqs_data) and not(te_data):
        return out_maps
        

    