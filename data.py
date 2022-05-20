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

def load_hdf5(ds_dir,hdf5_file,ech_idx):
    f = h5py.File(dataset_dir + hdf5_file, 'r')
    acqs = f1['Acquisitions'][...]
    out_maps = f1['OutMaps'][...]
    f.close()

    idxs_list = []
    for nd in range(len(acqs)):
        if np.sum(acqs[nd,:,:,1])!=0.0:
            idxs_list.append(nd)

    acqs = acqs[idxs_list,:,:,:ech_idx]
    out_maps = out_maps[idxs_list,:,:,:]

    return acqs, out_maps