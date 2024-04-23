import os
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _extract_fn(tfrecord):
    # Extract features using the keys set during creation
    features = {'acqs': tf.io.FixedLenFeature([], tf.string),
    			'acq_shape': tf.io.FixedLenFeature([5], tf.int64, default_value=[0,0,0,0,0]),
    			'out_maps': tf.io.FixedLenFeature([], tf.string),
    			'out_shape': tf.io.FixedLenFeature([5], tf.int64, default_value=[0,0,0,0,0])}
    
    # features = {"wf_real": _float_feature(wf_real.flatten()),
    #             "wf_imag": _float_feature(wf_imag.flatten()),
    #             "r2": _float_feature(r2.flatten()),
    #             "fm": _float_feature(fm.flatten()),
    #             'height': _int64_feature(wf.shape[0]),
    #             'width': _int64_feature(wf.shape[1]),
    #             'num_slices': _int64_feature(wf.shape[2]),
    #             'num_specie': _int64_feature(wf.shape[3])}

    # Extract the data record
    sample = tf.io.parse_single_example(tfrecord, features)

    acqs = tf.io.decode_raw(sample['acqs'], tf.float32)
    acq_shape = sample["acq_shape"]
    out_maps = tf.io.decode_raw(sample['out_maps'], tf.float32)
    out_shape = sample["out_shape"]
    
    return [acqs, acq_shape, out_maps, out_shape]


def extract_image(iterator, batch_size, angles= 0, training= False):

    one_data = iterator.get_next()
    '''
    one_data is a list of tensor contains [image,  label, img_shape, filename]
    The function returns a list of [augmentated image, label, image_shape, filename]
    '''
    acqs = one_data[0]
    acq_shape = one_data[1]
    out_maps = one_data[2] 
    out_shape = one_data[3] 

    #convert image,  label from 1D arrays to 2D arrays
    acqs = tf.reshape(acqs, shape=acq_shape, name=None)
    # sst = tf.image.resize(image, size  = Image_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,antialias=False, name=None)
    out_maps = tf.reshape(out_maps, shape=out_shape, name=None)
    
    #If you want to augment the images ( here augmentation has been performed by rotating images with some angle )
    # if training == True:
    #     image , label = image_augmentation2(image, label, angles)

    return acqs, out_maps


def extract_image2(iterator):
	return iterator.get_next()


recordPath = "tfrecord/LDM_ds"
dataset = tf.data.TFRecordDataset([recordPath])
# dataset.batch(2)

# Create a description of the features.
feature_description = {
	'acqs': tf.io.FixedLenFeature([], tf.string),
    'acq_shape': tf.io.FixedLenFeature([4], tf.int64),
    'out_maps': tf.io.FixedLenFeature([], tf.string),
    'out_shape': tf.io.FixedLenFeature([4], tf.int64),
}

def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    parsed_ds = tf.io.parse_example(example_proto, feature_description)
    return tf.io.parse_tensor(parsed_ds['acqs'], out_type=tf.float32), tf.io.parse_tensor(parsed_ds['out_maps'], out_type=tf.float32)

parsed_dataset = dataset.map(_parse_function)
parsed_dataset = parsed_dataset.batch(6).shuffle(36)

print('Dataset length:',parsed_dataset.cardinality().numpy())

i=1
for A, B in parsed_dataset:
    print(type(A))
    # A = list()
    # B = list()
    # for j in range(len(AB['acqs'])):
    #     A_j = tf.io.parse_tensor(AB['acqs'][j], out_type=tf.float32)
    #     A.append(tf.expand_dims(A_j,axis=0))
    #     B_j = tf.io.parse_tensor(AB['out_maps'][j], out_type=tf.float32)
    #     B.append(tf.expand_dims(B_j,axis=0))
    # A = tf.concat(A,axis=0)
    # B = tf.concat(B,axis=0)
    # A = tf.io.parse_tensor(A, out_type=tf.float32)
    print(str(i), A.shape, B.shape)
    i+=1

