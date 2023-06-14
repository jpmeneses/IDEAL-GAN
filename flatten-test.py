import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def encoder(input_shape):
	x = inputs = tf.keras.Input(input_shape)
	x = tf.keras.layers.Flatten()(x)
	encoded_size = x.shape[-1]
	std = tf.ones_like(x,dtype=tf.float32)*1e0
	x = tf.keras.layers.concatenate([x,std],axis=-1)
	# x = tf.keras.layers.Lambda(lambda x: tf.stack([x,tf.ones_like(x)*1e-7],2))(x)
	# x = tf.keras.layers.Reshape(target_shape=(2*encoded_size,))(x)
	x = tfp.layers.IndependentNormal(input_shape)(x)
	return tf.keras.Model(inputs=inputs,outputs=x)

hgt,wdt,n_ch = 24,24,1
enc = encoder((hgt,wdt,n_ch))
enc.summary()

def decoder(input_shape):
	x = inputs = tf.keras.Input(input_shape)
	x = tf.keras.layers.Reshape(target_shape=(24,24,1))(x)
	return tf.keras.Model(inputs=inputs,outputs=x)

dec = decoder((hgt*wdt*n_ch))

G_A2B = tf.keras.Sequential()
G_A2B.add(enc)
#G_A2B.add(dec)

A_sc = 1e1
A_zero_horiz = tf.zeros((1,hgt//3,wdt,1),dtype=tf.float32)
A_zero = tf.zeros((1,hgt//3,wdt//3,1),dtype=tf.float32)
A_one = tf.ones((1,hgt//3,wdt//3,1),dtype=tf.float32)*3*A_sc/4
A_aux = tf.concat([A_zero,A_one,A_zero],axis=2)
A = tf.concat([A_zero_horiz,A_aux,A_zero_horiz],axis=1)
B = G_A2B(A)

print('A shape:',A.shape)
print('B shape:',B.shape)

fig, axs = plt.subplots(figsize=(6, 3), nrows=1, ncols=2)
A_plt = axs[0].imshow(tf.squeeze(A), cmap='twilight', vmin=-A_sc, vmax=A_sc)
fig.colorbar(A_plt, ax=axs[0])
axs[0].axis('off')
B_plt = axs[1].imshow(tf.squeeze(B), cmap='twilight', vmin=-A_sc, vmax=A_sc)
fig.colorbar(B_plt, ax=axs[1])
axs[1].axis('off')
plt.show()