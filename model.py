from BatchLoader import *
import numpy as np
import tensorflow as tf
import cv2
import time
from math import floor

def get_placeholders(n_H0 = 341, n_W0 = 426, n_C0 = 1, n_y = 6):
	"""
	Inpus (Params):
		n_H - image height (in px) (defaults to 341)
		n_W - image width (in px) (defaults to 426)
		n_C - number of channels in image (defaults to 1 for "grayscale")
		n_y - number of network outputs (defaults to 6)

	Returns:
		X, y - Tensorflow placeholders for cnn input and labels.
	"""

	X = tf.placeholder(dtype = tf.float32, shape = [None, n_H0, n_W0, n_C0], name = "X_in")
	y = tf.placeholder(dtype = tf.float32, shape = [None, n_y], name = "y_true")

	return X, y

def init_params(n_H0 = 341, n_W0 = 426, n_C0 = 1, n_y = 6, check = False):
	"""
	Inpus (Params):
		n_H - image height (in px) (defaults to 341)
		n_W - image width (in px) (defaults to 426)
		n_C - number of channels in image (defaults to 1 for "grayscale")
		n_y - number of network outputs (defaults to 6)

	Returns:
		params, hparams - Parameters and Hyperparameters for the model.
	"""
	
	params = {}

	# CONV_11 ("VALID")
	n_C11 = 64
	f11 = 3	# filter_size
	s11 = 2	# stride
	p11 = 0	# padding
	W11 = tf.get_variable(dtype = tf.float32,
		shape = [f11, f11, n_C0, n_C11],
		initializer = tf.glorot_uniform_initializer(),
		name = "W11")
	n_H11 = floor((n_H0 + 2 * p11 - f11) / s11) + 1 # 170
	n_W11 = floor((n_W0 + 2 * p11 - f11) / s11) + 1 # 212
	if check:
		assert(n_H11 == 170 and n_W11 == 212)

	# CONV_12 ("VALID")
	n_C12 = 128
	f12 = 3	# filter_size
	s12 = 2	# stride
	p12 = 0	# padding
	W12 = tf.get_variable(dtype = tf.float32,
		shape = [f12, f12, n_C11, n_C12],
		initializer = tf.glorot_uniform_initializer(),
		name = "W12")
	
	n_H12 = floor((n_H11 + 2 * p12 - f12) / s12) + 1 # 84
	n_W12 = floor((n_W11 + 2 * p12 - f12) / s12) + 1 # 105
	if check:
		assert(n_H12 == 84 and n_W12 == 105)

	# POOL_1
	f_p1 = 2
	s_p1 = 2
	p_p1 = 0
	n_H12 = floor((n_H12 - f_p1) / s_p1) + 1 # 42
	n_W12 = floor((n_W12 - f_p1) / s_p1) + 1 # 52
	if check:
		assert(n_H12 == 42 and n_W12 == 52)

	# CONV_21 ("SAME")
	n_C21 = 128
	f21 = 3	# filter_size
	s21 = 1	# stride
	p21 = 1	# padding for SAME CONV
	W21 = tf.get_variable(dtype = tf.float32,
		shape = [f21, f21, n_C12, n_C21],
		initializer = tf.glorot_uniform_initializer(),
		name = 'W21')
	
	n_H21 = floor((n_H12 + 2 * p21 - f21) / s21) + 1 # 42
	n_W21 = floor((n_W12 + 2 * p21 - f21) / s21) + 1 # 52
	if check:
		assert(n_H21 == 42 and n_W21 == 52)		

	# CONV_22 ("SAME")
	n_C22 = 256
	f22 = 5	# filter_size
	s22 = 1	# stride
	p22 = 2	# padding for SAME CONV
	W22 = tf.get_variable(dtype = tf.float32,
		shape = [f22, f22, n_C21, n_C22],
		initializer = tf.glorot_uniform_initializer(),
		name = 'W22')
	
	n_H22 = floor((n_H21 + 2 * p22 - f22) / s22) + 1 # 42
	n_W22 = floor((n_W21 + 2 * p22 - f22) / s22) + 1 # 52
	if check:
		assert(n_H22 == 42 and n_W22 == 52)

	# POOL_2
	f_p2 = 2
	s_p2 = 2
	p_p2 = 0
	n_H22 = floor((n_H22 - f_p2) / s_p2) + 1 # 21
	n_W22 = floor((n_W22 - f_p2) / s_p2) + 1 # 26
	if check:
		assert(n_H22 == 21 and n_W22 == 26)

	"""
	INPUT FOR FC LAYERS: 21 x 26 x 256
	"""

	params["W11"] = W11
	params["W12"] = W12
	params["W21"] = W21
	params["W22"] = W22

	hparams = {"CONV_11":(s11, p11),
	"CONV_12":(s12, p12),
	"POOL_1":(f_p1, s_p1, p_p1),
	"CONV_21":(s21, p21),
	"CONV_22":(s22, p22),
	"POOL_2":(f_p2, s_p2, p_p2)}

	return params, hparams

params, hparams = init_params()
W11 = params["W11"]
W12 = params["W12"]
W21 = params["W21"]
W22 = params["W22"]

s11, p11 = hparams["CONV_11"]
s12, p12 = hparams["CONV_12"]
f_p1, s_p1, p_p1 = hparams["POOL_1"]
s21, p21 = hparams["CONV_21"]
s22, p22 = hparams["CONV_22"]
f_p2, s_p2, p_p2 = hparams["POOL_2"]

def normalize_input(tensor):
	"""
	Input (params):
	tensor - Input tensor to normalize.
	
	Outputs:
	tensor, element-wise divided by 255.
	"""
	return tf.divide(tensor, 255.)

def forward_prop(X, n_y = 6, training = True): # Use of 'training' parameter will be made when implementing batch_norm using the layers API.
	"""
	Inputs (params):
	X - The tensorflow input placeholder of appropriate shape. See docstring for 'get_placeholders'.
	training - Boolean, represents whether forward_prop is for training or inference.

	Outputs:
	fc3 - Network output tensor.
	"""
	X_norm =  normalize_input(X)

	Z11 = tf.nn.conv2d(input = X_norm,
		filter = W11,
		strides = [1, s11, s11, 1],
		padding = "VALID",
		data_format = "NHWC",
		name = "Z11")
	# Insert BN layer here.
	A11 = tf.nn.relu(Z11, name = "A11")

	# conv11 = tf.layers.conv2d(inputs = X,
	# 	filters = 64,
	# 	kernel_size = [3, 3],
	# 	strides = 2,
	# 	padding = "valid",
	# 	data_format = "channels_last",
	# 	activation = tf.nn.relu,
	# 	name = "CONV_1.1")

	Z12 = tf.nn.conv2d(input = A11,
		filter = W12,
		strides = [1, s12, s12, 1],
		padding = "VALID",
		data_format = "NHWC",
		name = "Z12")
	# Insert BN layer here.
	A12 = tf.nn.relu(Z12, name = "A12")

	# conv12 = tf.layers.conv2d(inputs = conv11,
	# 	filters = 128,
	# 	kernel_size = [3, 3],
	# 	strides = 2,
	# 	padding = "valid",
	# 	data_format = "channels_last",
	# 	activation = tf.nn.relu,
	# 	name = "CONV_1.2")

	P1 = tf.nn.max_pool(value = A12,
		ksize = [1, f_p1, f_p1, 1],
		strides = [1, s_p1, s_p1, 1],
		padding = "VALID",
		data_format = "NHWC",
		name = "P1")

	# pool1 = tf.layers.max_pooling2d(inputs = conv12,
	# 	pool_size = [2, 2],
	# 	strides = 2,
	# 	padding = 'valid',
 	#	data_format = 'channels_last',
 	#	name = "POOL_1")

	Z21 = tf.nn.conv2d(input = P1,
		filter = W21,
		strides = [1, s21, s21, 1],
		padding = "SAME",
		data_format = "NHWC",
		name = "Z21")
	# Insert BN layer here.
	A21 = tf.nn.relu(Z21, name = "A21")

	# conv21 = tf.layers.conv2d(inputs = pool1,
	# 	filters = 128,
	# 	kernel_size = [3, 3],
	# 	strides = 1,
	# 	padding = "same",
	# 	data_format = "channels_last",
	# 	activation = tf.nn.relu,
	# 	name = "CONV_2.1")

	Z22 = tf.nn.conv2d(input = A21,
		filter = W22,
		strides = [1, s22, s22, 1],
		padding = "SAME",
		data_format = "NHWC",
		name = "Z22")
	# Insert BN layer here.
	A22 = tf.nn.relu(Z22, name = "A22")

	# conv22 = tf.layers.conv2d(inputs = conv21,
	# 	filters = 256,
	# 	kernel_size = [5, 5],
	# 	strides = 1,
	# 	padding = "same",
	# 	data_format = "channels_last",
	# 	activation = tf.nn.relu,
	# 	name = "CONV_2.2")

	P2 = tf.nn.max_pool(value = A22,
		ksize = [1, f_p2, f_p2, 1],
		strides = [1, s_p2, s_p2, 1],
		padding = "VALID",
		data_format = "NHWC",
		name = "P2")

	# pool2 = tf.layers.max_pooling2d(inputs = conv22,
	# 	pool_size = [2, 2],
	# 	strides = 2,
	# 	padding = 'valid',
 # 		data_format = 'channels_last',
 # 		name = "POOL_2")

	fc_input = tf.layers.flatten(inputs = P2,
		name = "FLATTEN")

	assert(fc_input.shape[1] == 21 * 26 * 256)

	fc1 = tf.layers.dense(inputs = fc_input,
		units = 256, # ORIGIALLY 4096, REDUCED DUE TO OOM ResourceExhaustError.
		activation = tf.nn.relu,
		name = "FC_1-4096")

	fc2 = tf.layers.dense(inputs = fc1,
		units = 512,
		activation = tf.nn.relu,
		name = "FC_2-512")
	fc3 = tf.layers.dense(inputs = fc2,
		units = n_y,
		activation = None, # activation = None implies a linear activation.
		name = "FC_3-n_y")

	return fc3



def model(minibatch_size = 32, num_epochs = 8, learning_rate = 0.001, mode = "TRAIN"):

	X, y = get_placeholders() # Include input parameters in the function call for different input dims.

	predictions = forward_prop(X)

	# loss = custom_loss_placeholder_function()

	loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	init = tf.global_variables_initializer()

	# Saver init goes here.

	batch_gen = BatchLoader(batch_size = 2048,
			minibatch_size = minibatch_size,
			num_data_points = 22528,
			cwd = "./",
			image_folder_relative = "IMAGE_DATA_2",
			ew_file = "_elbow_wrist.txt",
			wh_file = "_wrist_hand.txt")

	if mode == "TRAIN":
		loss_val = 0
		with tf.Session() as sess:
			sess.run(init)
			# saver.restore(...)
			printed_once = False
			batch_ctr = 0
			while batch_gen.epochs_passed != num_epochs:
				X_buffer, y_buffer = batch_gen.get_next_batch()
				m_batch = X_buffer.shape[0]
				num_slices = m_batch // minibatch_size
				# if m_batch / minibatch_size == num_slices:
				# 	for i in range(num_slices):
				# 		start = i * minibatch_size
				# 		end = start + minibatch_size
				# 		X = X_buffer[start:end]
				# 		y = y_buffer[start:end]
				batch_loss = 0
				tick = time.time()
				for i in range(num_slices): # *IMPORTANT* - THIS WILL NEGLECT THE LAST 'NUM_DATA_POINT % MINIBATCH_SIZE' EXAMPLES.
					start = i * minibatch_size
					end = start + minibatch_size
					X_mb = X_buffer[start:end]
					y_mb = y_buffer[start:end]
					_, loss_val = sess.run([optimizer, loss], feed_dict = {X:X_mb, y:y_mb})
					batch_loss += (loss_val / num_slices)
				tock = time.time()

				# Code for printing out the correct zero-indexed batch number just processed.
				if batch_gen.current_batch != 0:
					batch_ctr = batch_gen.current_batch - 1
				else:
					batch_ctr = 10

				# Code for printing current satus of training incl. averaged batch loss, current batch passed, number of epochs passed.
				# if (batch_gen.current_batch == 0) and (batch_gen.epochs_passed % 1 == 0):

				# 	print("Epochs passed: " + str(batch_gen.epochs_passed) + "   |   Batch Processed: " + str(batch_ctr))
				# 	print("LOSS: " + batch_loss)
				print("Epochs passed: " + str(batch_gen.epochs_passed) + " | Batch processed: " + str(batch_ctr) + " | Batch train time: " + str(tock - tick) + " sec." + " | Loss: " + str(batch_loss))
				
				# print("Loss: " + batch_loss)				

				
	if mode == "VALIDATE":
		loss_val = 0
		validation_batch_loss = 0
		with tf.Session() as sess:
			sess.run(init)
			"""****************BEGIN HERE ON 18th June 2018****************"""
			X_val, y_val = batch_gen.get_validation_batch(remove_stray = True) # Remember to change this function's implementation to use unseen examples for validation
			print(X_val.shape[0], y_val.shape[0])
			assert(X_val.shape[0] == y_val.shape[0])
			m_val_batch = X_val.shape[0]
			num_slices = m_val_batch // minibatch_size
			print(batch_gen.total_num_data_points, batch_gen.num_data_points)
			print(num_slices, m_val_batch / minibatch_size)
			assert(num_slices == m_val_batch / minibatch_size) # Just to check/prevent the presence of stray images.
			tick = time.time()
			for i in range(num_slices): # *IMPORTANT* - THIS WILL NEGLECT THE LAST 'NUM_DATA_POINT % MINIBATCH_SIZE' EXAMPLES.
				start = i * minibatch_size
				end = start + minibatch_size
				X_mb = X_val[start:end]
				y_mb = y_val[start:end]
				loss_val = sess.run(loss, feed_dict = {X:X_mb, y:y_mb})
				validation_batch_loss += (loss_val / num_slices)
			tock = time.time()
			print("Validation batch size: " + str(m_val_batch) + " | Validation forward propagation time: " + str(tock - tick) + " sec." + " | Loss: " + str(validation_batch_loss))




	pass

def main():
	# model(minibatch_size = 8, num_epochs = 8, learning_rate = 0.001, mode = "TRAIN")
	model(minibatch_size = 8, num_epochs = 8, learning_rate = 0.001, mode = "VALIDATE")

if __name__ == '__main__':
	main()