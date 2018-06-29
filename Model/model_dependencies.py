from BatchLoader import *
import numpy as np
import tensorflow as tf
import cv2
import time
from math import floor

def variable_summaries(var):
	"""Attach different summaries to a Tensor (for TensorBoard visualization)."""
	
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    	tf.summary.scalar('stddev', stddev)
	    tf.summary.histogram('histogram', var)

def log_all_var():
	"""To attach variables to the TensorBoard summary all at once."""
	variable_summaries(tf.get_variable('W11:0'))
	variable_summaries(tf.get_variable('W12:0'))
	variable_summaries(tf.get_variable('W21:0'))
	variable_summaries(tf.get_variable('W22:0'))
	variable_summaries(tf.get_variable('Z11_BN/gamma:0'))
	variable_summaries(tf.get_variable('Z11_BN/beta:0'))
	variable_summaries(tf.get_variable('Z12_BN/gamma:0'))
	variable_summaries(tf.get_variable('Z12_BN/beta:0'))
	variable_summaries(tf.get_variable('Z21_BN/gamma:0'))
	variable_summaries(tf.get_variable('Z21_BN/beta:0'))
	variable_summaries(tf.get_variable('Z22_BN/gamma:0'))
	variable_summaries(tf.get_variable('Z22_BN/beta:0'))
	variable_summaries(tf.get_variable('FC_1-4096/kernel:0'))
	variable_summaries(tf.get_variable('FC_1-4096/bias:0'))
	variable_summaries(tf.get_variable('FC_2-512/kernel:0'))
	variable_summaries(tf.get_variable('FC_2-512/bias:0'))
	variable_summaries(tf.get_variable('FC_3-n_y/kernel:0'))
	variable_summaries(tf.get_variable('FC_3-n_y/bias:0'))

def get_placeholders(n_H0 = 170, n_W0 = 213, n_C0 = 3, n_y = 6): # Changed n_C0 to 3 (color images) (initially was 1)
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

def init_params(n_H0 = 170, n_W0 = 213, n_C0 = 3, n_y = 6, check = False): # Changed n_C0 to 3 (color images) (initially was 1)
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
	# variable_summaries(W11) # For tensorboard logging.

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
	# variable_summaries(W12) # For tensorboard logging.

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
	# variable_summaries(W21) # For tensorboard logging.
	
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
	# variable_summaries(W22) # For tensorboard logging.
	
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

def normalize_input(tensor):
	"""
	Input (params):
	tensor - Input tensor to normalize.
	
	Outputs:
	tensor, element-wise divided by 255.
	"""
	return tf.divide(tensor, 255.)

def forward_prop(X, n_y = 6, training = False): # Use of 'training' parameter will be made when implementing batch_norm using the layers API.
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
	# Z11_BN = tf.layers.batch_normalization(inputs = Z11, axis = 3, training = training, name = "Z11_BN")
	A11 = tf.nn.relu(Z11, name = "A11") # _BN, name = "A11")

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
	# Z12_BN = tf.layers.batch_normalization(inputs = Z12, axis = 3, training = training, name = "Z12_BN")
	A12 = tf.nn.relu(Z12, name = "A12") # _BN, name = "A12")

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
	# Z21_BN = tf.layers.batch_normalization(inputs = Z21, axis = 3, training = training, name = "Z21_BN")
	A21 = tf.nn.relu(Z21, name = "A21") # _BN, name = "A21")

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
	# Z22_BN = tf.layers.batch_normalization(inputs = Z22, axis = 3, training = training, name = "Z22_BN")
	A22 = tf.nn.relu(Z22, name = "A22") # _BN, name = "A22")

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

	# assert(fc_input.shape[1] == 21 * 26 * 256)

	fc1 = tf.layers.dense(inputs = fc_input,
		units = 1024, # ORIGIALLY 4096, REDUCED DUE TO OOM ResourceExhaustError.
		activation = tf.nn.relu,
		name = "FC_1-4096")
	fc1_drop = tf.layers.dropout(inputs = fc1, rate = 0.0, training = training, name = "fc1_drop")

	fc2 = tf.layers.dense(inputs = fc1_drop,
		units = 4096,
		activation = tf.nn.relu,
		name = "FC_2-512")
	fc2_drop = tf.layers.dropout(inputs = fc2, rate = 0.0, training = training, name = "fc2_drop")

	fc3 = tf.layers.dense(inputs = fc2_drop,
		units = n_y,
		activation = None, # activation = None implies a linear activation.
		name = "FC_3-n_y")

	return fc3

def custom_loss(pred, y, k = 0.85): # Remember to change k to 0.85 because the unit used is metres and not centimetres.
	x1_p, y1_p, z1_p, x2_p, y2_p, z2_p = tf.split(pred, num_or_size_splits = 6, axis = 1)
	x1_y, y1_y, z1_y, x2_y, y2_y, z2_y = tf.split(y, num_or_size_splits = 6, axis = 1)

	x_p = ((-1 * (y1_p + k) * (x2_p - x1_p)) / (y2_p - y1_p)) + x1_p
	z_p = ((-1 * (y1_p + k) * (z2_p - z1_p)) / (y2_p - y1_p)) + z1_p

	x_y = ((-1 * (y1_y + k) * (x2_y - x1_y)) / (y2_y - y1_y)) + x1_y
	z_y = ((-1 * (y1_y + k) * (z2_y - z1_y)) / (y2_y - y1_y)) + z1_y

	return tf.reduce_mean(tf.add(tf.square(tf.subtract(x_p, x_y)), tf.square(tf.subtract(z_p, z_y))))

