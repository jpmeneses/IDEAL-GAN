import os
import json
import numpy as np
import tensorflow as tf
import h5py
from skimage.restoration import unwrap_phase

import pylib as py

import nibabel as nib
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids

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


def unwrap_slices(x):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i,:,:] = unwrap_phase(x[i,:,:],wrap_around=False)
    return np.expand_dims(y,axis=-1)


def load_hdf5(ds_dir,hdf5_file,ech_idx=12,start=0,end=2000,custom_list=None,num_slice_list=None,
            remove_non_central=False,acqs_data=True,te_data=False,complex_data=False,
            remove_zeros=True, MEBCRN=False, mag_and_phase=False, unwrap=False):
    f = h5py.File(ds_dir + hdf5_file, 'r')
    if custom_list:
        if acqs_data:
            acqs = f['Acquisitions'][custom_list]
        out_maps = f['OutMaps'][custom_list]
        if te_data:
            TEs = f['TEs'][custom_list]
            TEs = np.expand_dims(TEs,axis=-1)
    elif num_slice_list:
        ini_end_idxs = np.cumsum(num_slice_list)
        # ini_end_idxs = np.concatenate((np.zeros((1),dtype=int),ini_end_idxs))
        idxs = list()
        for k in range(ini_end_idxs[0],ini_end_idxs[-1]):
            k_diff = k-ini_end_idxs[0]
            if np.abs(k_diff) > 4:
                idxs.append(k)
            elif k_diff >= 4:
                ini_end_idxs = np.delete(ini_end_idxs,0)
        if acqs_data:
            acqs = f['Acquisitions'][idxs]
        out_maps = f['OutMaps'][idxs]
        if te_data:
            TEs = f['TEs'][idxs]
            TEs = np.expand_dims(TEs,axis=-1)
    else:
        if acqs_data:
            acqs = f['Acquisitions'][start:end]
        out_maps = f['OutMaps'][start:end]
        if te_data:
            TEs = f['TEs'][start:end]
            TEs = np.expand_dims(TEs,axis=-1)
    f.close()

    if remove_zeros:
        idxs_list = []
        for nd in range(len(out_maps)):
            if np.sum(out_maps[nd,:,:,0])!=0.0:
                idxs_list.append(nd)
    else:
        idxs_list = [i for i in range(len(out_maps))]

    # Pre-process out maps
    out_maps = out_maps[idxs_list,:,:,:]
    if MEBCRN:
        if mag_and_phase: # SHAPE: [BS,2,H,W,Nmaps]
            out_w_mag = np.sqrt(np.sum(out_maps[:,:,:,:2]**2,axis=-1,keepdims=True)) # [BS,H,W,1]
            out_f_mag = np.sqrt(np.sum(out_maps[:,:,:,2:4]**2,axis=-1,keepdims=True)) # [BS,H,W,1]
            out_ff = np.divide(out_f_mag, out_w_mag+out_f_mag, where=(out_w_mag+out_f_mag)!=0.0) # [BS,H,W,1]
            out_ff = np.expand_dims(np.concatenate((out_ff,np.zeros_like(out_ff)),axis=-1), axis=1) # [BS,1,,H,W,2]
            out_pd = out_w_mag+out_f_mag # [BS,H,W,1]
            out_mag = np.expand_dims(np.concatenate((out_pd,out_maps[:,:,:,4:5]),axis=-1),axis=1) # [BS,1,H,W,2]
            out_w_pha = np.where(out_w_mag>0,np.arctan2(out_maps[:,:,:,1:2],out_maps[:,:,:,0:1]),0.0) # [BS,H,W,1]
            out_f_pha = np.where(out_f_mag>0,np.arctan2(out_maps[:,:,:,3:4],out_maps[:,:,:,2:3]),0.0) # [BS,H,W,1]
            out_wf_pha = np.divide(out_w_mag*out_w_pha+out_f_mag*out_f_pha, out_w_mag+out_f_mag, where=(out_w_mag+out_f_mag)!=0.0)
            if unwrap:
                out_wf_pha = unwrap_slices(np.squeeze(out_wf_pha))
                k_phase = 4*np.pi
            else:
                k_phase = np.pi
            out_pha = np.expand_dims(np.concatenate((out_wf_pha/k_phase,out_maps[:,:,:,5:]),axis=-1),axis=1) # [BS,1,H,W,2]
            out_maps = np.concatenate((out_ff,out_mag,out_pha),axis=1)
            ns,_,hgt,wdt,_ = out_maps.shape
        else: # SHAPE: [BS,Nmaps,H,W,2]
            out_rho_w = np.expand_dims(out_maps[:,:,:,:2],axis=1)
            out_rho_f = np.expand_dims(out_maps[:,:,:,2:4],axis=1)
            out_xi = np.concatenate((out_maps[:,:,:,5:],out_maps[:,:,:,4:5]),axis=-1)
            out_xi = np.expand_dims(out_xi,axis=1)
            out_maps = np.concatenate((out_rho_w,out_rho_f,out_xi),axis=1)
            ns,_,hgt,wdt,_ = out_maps.shape
    else:
        ns,hgt,wdt,_ = out_maps.shape

    if acqs_data:
        acqs = acqs[idxs_list,:,:,:ech_idx]
        if complex_data or MEBCRN:
            acqs_real = acqs[:,:,:,0::2]
            acqs_imag = acqs[:,:,:,1::2]
            if complex_data:
                acqs = acqs_real + 1j*acqs_imag
            elif MEBCRN:
                acqs = np.zeros([len(out_maps),acqs_real.shape[-1],hgt,wdt,2],dtype='single')
                acqs[:,:,:,:,0] = np.transpose(acqs_real,(0,3,1,2))
                acqs[:,:,:,:,1] = np.transpose(acqs_imag,(0,3,1,2))
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
        

def group_TEs(A,B,TEs,TE1,dTE,MEBCRN=False):
    TE1 = np.float32(TE1)
    dTE = np.float32(dTE)
    TE1_orig = np.float32(0.0013)
    dTE_orig = np.float32(0.0021)

    if MEBCRN:
        len_dataset,ne,hgt,wdt,_ = A.shape
        _,n_out,_,_,_ = B.shape
    else:
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
            TE1_i = np.round(TEs[idx,0,0],4)
            dTE_i = np.round(np.mean(np.diff(TEs[idx:idx+1,:,0])),4)
        else:
            TE1_i = TE1_orig
            dTE_i = dTE_orig
        # print(idx,TE1_i,TE1_orig,'\t',dTE_i,dTE_orig)

        # Check if it corresponds to original TEs
        if TE1_i==TE1_orig and dTE_i==dTE_orig:
            if not(flag_orig): # First orig TEs' slice
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

    if MEBCRN:
        A[all_null_slices,:,:,:,:] = 0.0
        B[all_null_slices,:,:,:,:] = 0.0
    else:
        A[all_null_slices,:,:,:] = 0.0
        B[all_null_slices,:,:,:] = 0.0

    if MEBCRN:
        A = A[all_sel_slices,:,:,:,:]
        B = B[all_sel_slices,:,:,:,:]
    else:
        A = A[all_sel_slices,:,:,:]
        B = B[all_sel_slices,:,:,:]
    TEs = TEs[all_sel_slices,:]

    return A, B, TEs


def A_from_MEBCRN(A):
    A_r = A[...,0]
    A_i = A[...,1]

    A_r = tf.transpose(A_r,perm=[0,2,3,1])
    A_i = tf.transpose(A_i,perm=[0,2,3,1])
    nb,hgt,wdt,ne = A_r.shape

    zero_fill = tf.zeros_like(A_r)
    re_stack_var = tf.stack([A_r,zero_fill],4)
    re_aux_var = tf.reshape(re_stack_var,[nb,hgt,wdt,2*ne])
    im_stack_var = tf.stack([zero_fill,A_i],4)
    im_aux_var = tf.reshape(im_stack_var,[nb,hgt,wdt,2*ne])

    return re_aux_var + im_aux_var


def B_from_MEBCRN(B,mag_and_phase=False,c_pha=3):
    if mag_and_phase:
        B_W_r = B[:,0,:,:,:1] * tf.math.cos(c_pha*B[:,1,:,:,1:2]*np.pi)
        B_W_i = B[:,0,:,:,:1] * tf.math.sin(c_pha*B[:,1,:,:,1:2]*np.pi)
        B_F_r = B[:,0,:,:,1:2]* tf.math.cos(c_pha*B[:,1,:,:,1:2]*np.pi)
        B_F_i = B[:,0,:,:,1:2]* tf.math.sin(c_pha*B[:,1,:,:,1:2]*np.pi)
        B_r2 = B[:,0,:,:,2:]
        B_fm = B[:,1,:,:,2:]
        return tf.concat([B_W_r,B_W_i,B_F_r,B_F_i,B_r2,B_fm],axis=-1)
    else:
        B_W = B[:,0,:,:,:]
        B_F = B[:,1,:,:,:]
        B_PM= B[:,2,:,:,:]
        B_R2= B_PM[:,:,:,1:]
        B_FM= B_PM[:,:,:,:1]
        return tf.concat([B_W,B_F,B_R2,B_FM],axis=-1)


def B_to_MEBCRN(B,mode='All'):
    if mode == 'WF':
        B_W = tf.expand_dims(B[:,:,:,:1],axis=1)
        B_W = tf.concat([B_W,tf.zeros_like(B_W)],axis=-1)
        B_F = tf.expand_dims(B[:,:,:,1:],axis=1)
        B_F = tf.concat([B_F,tf.zeros_like(B_F)],axis=-1)
        return tf.concat([B_W,B_F],axis=1)
    elif mode == 'PM':
        B_R2 = tf.expand_dims(B[:,:,:,:1],axis=1)
        B_PM = tf.expand_dims(B[:,:,:,1:],axis=1)
        return tf.concat([B_PM,B_R2],axis=-1)
    elif mode == 'WF-PM':
        B_W = B[:,:,:,:1]
        B_W = tf.concat([B_W,tf.zeros_like(B_W)],axis=-1)
        B_F = B[:,:,:,1:2]
        B_F = tf.concat([B_F,tf.zeros_like(B_F)],axis=-1)
        B_R2= B[:,:,:,2:3]
        B_FM= B[:,:,:,3:]
        B_PM= tf.concat([B_FM,B_R2],axis=-1)
        B_W = tf.expand_dims(B_W, axis=1)
        B_F = tf.expand_dims(B_F, axis=1)
        B_PM= tf.expand_dims(B_PM,axis=1)
        return tf.concat([B_W,B_F,B_PM],axis=1)
    elif mode == 'All':
        B_W = B[:,:,:,:2]
        B_F = B[:,:,:,2:4]
        B_R2= B[:,:,:,4:5]
        B_FM= B[:,:,:,5:]
        B_PM= tf.concat([B_FM,B_R2],axis=-1)
        B_W = tf.expand_dims(B_W, axis=1)
        B_F = tf.expand_dims(B_F, axis=1)
        B_PM= tf.expand_dims(B_PM,axis=1)
        return tf.concat([B_W,B_F,B_PM],axis=1)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


#################################################################################
############################ DICOM GEN FUNCTIONS ################################
#################################################################################

def gen_ds(idx,method_prefix='m000',R2s=False):
    # DICOM constant information
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID= pydicom.uid.ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
    ds.PatientName = "Volunteer^" + str(idx).zfill(3) + "^-" + method_prefix
    ds.PatientID = str(idx).zfill(6) # "123456"

    ds.Modality = "MR"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    ## These are the necessary imaging components of the FileDataset object.
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.ImagePositionPatient = r"0\0\1"
    ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "0"
    if R2s:
        ds.RescaleSlope = "0.78"
    else:
        ds.RescaleSlope = "0.4"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    return ds


def write_dicom(ds, pixel_array, path, filename, level, slices):
    image2d = np.squeeze(pixel_array)*255
    image2d = image2d.astype(np.uint16)

    # path = py.join(args.experiment_dir,"out_dicom",args.map,'Volunteer-'+nvol[1:],'Method-'+meth[1:])
    # py.mkdir(path)
    suffix = "_s" + str(level).zfill(2) + ".dcm"

    filename_endian= py.join(path, filename + suffix)
    ds.ImagesInAcquisition = str(slices)
    ds.InstanceNumber = level

    ds.Columns = image2d.shape[0]
    ds.Rows = image2d.shape[1]
    ds.PixelData = image2d.tobytes()

    ds.save_as(filename_endian, write_like_original=True)
    return


def load_dicom_series(folder_path):
    """
    Load DICOM series from a folder (multiple echoes per sample).
    
    Args:
        folder_path: str, path to folder containing DICOM files
        num_echoes: int, expected number of echoes (optional, for reshape)
        target_shape: tuple (H, W), desired image size (optional, for resize)
    
    Returns:
        np.ndarray of shape (num_echoes, H, W)
    """
    dicom_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.endswith(".dcm")])
    
    images = []
    n_sl = 0
    err_flag = False
    for j, f in enumerate(dicom_files):
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)

        img_comp = str(ds.get((0x2005, 0x1011), pydicom.DataElement((0x2005, 0x1011), 'DS', 1)).value)
        echo_time = float(ds.get((0x0018, 0x0081), pydicom.DataElement((0x0018, 0x0081), 'DS', 1)).value)
        echo_num = int(ds.get((0x0018, 0x0086), pydicom.DataElement((0x0018, 0x0086), 'DS', 1)).value)
        echo_all = int(ds.get((0x0018, 0x0091), pydicom.DataElement((0x0018, 0x0091), 'DS', 1)).value)
        RescaleIntercept = float(ds.get((0x2005, 0x100D), pydicom.DataElement((0x2005, 0x100D), 'DS', 1)).value)
        RescaleSlope = float(ds.get((0x2005, 0x100A), pydicom.DataElement((0x2005, 0x100A), 'DS', 1)).value)
        # print(j, "Component: ", img_comp)

        resc_img = RescaleSlope*(img-RescaleIntercept)

        if img_comp == "R" and echo_num == 1:
            err_flag = False
            echoes = list()

        if img_comp == "R":
            aux_img = np.expand_dims(resc_img,axis=-1)
        elif img_comp == "I":
            aux_img = np.concatenate([aux_img,np.expand_dims(resc_img,axis=-1)], axis=-1)
            if aux_img.shape[-1] == 2:
                echoes.append(aux_img)
            else:
                err_flag = True

            if echo_all == len(echoes) and (not err_flag):
                im_echoes = np.stack(echoes, axis=0)
                # Normalization considering max magnitude value
                mag_echoes = np.sqrt(np.sum(np.square(im_echoes),axis=-1,keepdims=True))
                im_echoes_norm = im_echoes/np.max(mag_echoes)
                # Masking considering magnitude threshold
                im_echoes_mean = np.mean(mag_echoes/np.max(mag_echoes),axis=0,keepdims=True)
                im_echoes_mask = np.repeat(im_echoes_mean>0.05,echo_all,axis=0)
                im_echoes_mask = np.repeat(im_echoes_mask,2,axis=-1)
                images.append(np.where(im_echoes_mask,im_echoes_norm,0.0))
                n_sl += 1

    # print("No. slices:", n_sl)
    # for i, arr in enumerate(images):
        # print(i, arr.shape, arr.dtype)
    images = np.stack(images, axis=0)  # (num_echoes, H, W, 2)

    # Quantitative maps
    # out_maps = np.ones([n_sl,3,images.shape[1],images.shape[2],2], dtype=np.float32)

    return images


def tf_load_dicom_series(folder_path):
    """
    Wrap load_dicom_series in tf.py_function for use in tf.data.Dataset.
    """
    def _load(path):
        path = path.numpy().decode("utf-8")
        arr = load_dicom_series(path)
        return arr

    data = tf.py_function(
        func=_load,
        inp=[folder_path],
        Tout=tf.float32
    )
    
    return data


def load_nifti_series(folder_path):
    dicom_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.endswith(".nii.gz")])
    avoid_comps = ['imaginary','real','Eq']
    dicom_files = sorted([f for f in dicom_files if not any(char in f for char in avoid_comps)])
    # print(dicom_files)

    nifti_file = dicom_files[0]
    fn_noEch = nifti_file.split("_e")[0]

    # Load NIfTI image
    img = nib.load(nifti_file)
    data = img.get_fdata(dtype=np.float32)  # shape: (X, Y, Z, echoes?) or (X, Y, Z)
    
    # Load JSON sidecar (contains echo times, etc.)
    json_file = nifti_file.replace(".nii.gz", ".json")
    metadata = {}
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            metadata = json.load(f)

    # Define useful parameters
    ne = metadata["EchoTrainLength"]

    # Generate multi-echo volume to be filled
    V_shape = list(data.shape)
    V_shape.insert(2, ne)
    V = np.zeros(V_shape+[2], dtype=np.float32)
    TE = np.zeros((V_shape[-1],ne), dtype=np.float32)
    V_mag_all = np.zeros(V_shape, dtype=np.float32)

    for ech in range(ne):
        # Magnitude file processing
        nifti_file_mag = fn_noEch + '_e' + str(ech+1) + '.nii.gz'
        img_mag = nib.load(nifti_file_mag)
        V_mag = img_mag.get_fdata(dtype=np.float32)

        json_file_mag = nifti_file_mag.replace(".nii.gz", ".json")
        metadata_mag = {}
        if os.path.exists(json_file_mag):
            with open(json_file_mag, "r") as f:
                metadata_mag = json.load(f)

        V_mag_resc = np.array(V_mag) # / float(metadata_mag["PhilipsScaleSlope"]);
        if ech == 0:
            V_sc = np.max(V_mag_resc)

        # Phase file processing
        nifti_file_ph = fn_noEch + '_e' + str(ech+1) + '_ph.nii.gz'
        img_ph = nib.load(nifti_file_ph)
        V_ph = img_ph.get_fdata(dtype=np.float32)

        json_file_ph = nifti_file_ph.replace(".nii.gz", ".json")
        metadata_ph = {}
        if os.path.exists(json_file_ph):
            with open(json_file_ph, "r") as f:
                metadata_ph = json.load(f)

        V_ph_resc = np.array(V_ph) # / float(metadata_ph["PhilipsScaleSlope"]);

        # Combining into complex volume
        if V_mag_resc.shape[2] == V_ph_resc.shape[2]:
            V_ech = V_mag_resc * np.exp(1j*V_ph_resc) / V_sc;
        else:
            V_ech = np.zeros([V.shape[0],V.shape[1],V.shape[3]]);
            print('\tMismatch between mag and phase at echo:',str(ech))

        if V.shape[3] == V_ech.shape[2]:
            V[:,:,ech,:,0] = np.real(V_ech);
            V[:,:,ech,:,1] = np.imag(V_ech);
            TE[:,ech] = float(metadata_mag["EchoTime"]);
            V_mag_all[:,:,ech,:] = np.abs(V_ech)
        else:
            print('\tMismatch between complex array and 1st echo at echo',str(ech))

    # Get mask from the mean of all-echoes magnitudes
    V_mag_mean = np.mean(V_mag_all,axis=2,keepdims=True)
    V_mag_mean = np.expand_dims(np.repeat(V_mag_mean,ne,axis=2),axis=-1)
    V_mag_mean = np.repeat(V_mag_mean,2,axis=-1)
    V = np.where(V_mag_mean >= 0.05, V, 0.0)
    
    # Rearrange to obtain the required dimensionality
    V = np.transpose(V, axes=[3,2,1,0,4]) # (num_slices, num_echoes, H, W, 2)
    V = np.flip(V,axis=2)

    return V[:,::2,...]


def tf_load_nifti_series(folder_path):
    """
    Wrap load_nifti_series in tf.py_function for use in tf.data.Dataset.
    """
    def _load(path):
        path = path.numpy().decode("utf-8")
        arr = load_nifti_series(path)
        return arr

    data = tf.py_function(
        func=_load,
        inp=[folder_path],
        Tout=tf.float32
    )
    
    return data