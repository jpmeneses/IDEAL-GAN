import tensorflow as tf
import tensorflow.keras as keras

vgg = keras.applications.vgg19.VGG19()

def metric_model(input_shape, layers=[2,5,8,13,18], pad=(16,16)):
	inputs = keras.Input(input_shape)
	output = list()
	for l in layers:
		metric_vgg = keras.Model(inputs=vgg.inputs, outputs=vgg.layers[l].output)
		x = keras.layers.Lambda(lambda x: tf.reshape(x,[-1,x.shape[2],x.shape[3],x.shape[4]]))(inputs)
		x = keras.layers.Lambda(lambda x: tf.concat([x,tf.math.sqrt(tf.reduce_sum(tf.math.square(x),axis=-1,keepdims=True))],axis=-1))(x)
		x = keras.layers.ZeroPadding2D(padding=pad)(x)
		x = metric_vgg(x)
		output.append(x)
	
	return keras.Model(inputs=inputs, outputs=output)


