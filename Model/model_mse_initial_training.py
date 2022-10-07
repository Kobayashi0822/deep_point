from batch_loader import *
from model_dependencies import *
import numpy as np
import tensorflow as tf
import cv2
import time
from math import floor

params, hparams = init_params() # Obtain all parameters and hyperparameters in appropriate dictionaries.

W11 = params["W11"]
W12 = params["W12"]
W21 = params["W21"]
W22 = params["W22"]

"""For tensorboard logging."""
variable_summaries(W11)
variable_summaries(W12)
variable_summaries(W21)
variable_summaries(W22)
"""------------------------"""

s11, p11 = hparams["CONV_11"]
s12, p12 = hparams["CONV_12"]
f_p1, s_p1, p_p1 = hparams["POOL_1"]
s21, p21 = hparams["CONV_21"]
s22, p22 = hparams["CONV_22"]
f_p2, s_p2, p_p2 = hparams["POOL_2"]

def model(minibatch_size = 32, num_epochs = 8, learning_rate = 0.001, mode = "TRAIN", restore_saved_model = False, save_every_n_epochs = 1, log_folder = '1'):
	"""
	Inputs (params):
	minibatch_size - integer, size of minibatches for each iteration of Adam.
	num_epochs - integer, number of training epochs
	learning_rate - float, learning_rate for Adam Optimizer.
	mode - string, represents whether model is being trained or validated. Either "TRAIN" or "VALIDATE"
	restore_saved_model - boolean, instucts the script to either initialize all parameters from scrath (False) or load a saved model (True).
	save_every_n_epochs - integer, number of epochs to pass before saving the current state of all model variables.
	log_folder - string, name of the folder for saving TensorBoard event logs for corresponding to the current run.

	Outputs:
	--None--
	"""

	X, y = get_placeholders() # Include input parameters in the function call for different input dims.

	if mode == "TRAIN":
		predictions = forward_prop(X, training = True)
	elif mode == "VALIDATE":
		predictions = forward_prop(X, training = False)
		
	predicted_points = tf.gather(predictions, list(range(4)))
	actual_points = tf.gather(y, list(range(4)))

	# loss = custom_loss_placeholder_function()

	loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)
	tf.summary.scalar('mse', loss) # For tensorboard logging.

	# Adding update_ops as a dependency to the optimizer for correct working of batch_normalization during training and inference
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

	init = tf.global_variables_initializer()

	# Saver init goes here.
	saver = tf.train.Saver() # Check if BN works fine with this not haveing any parameters. It might be possible that the moving averages are not being saved.
	save_path = "C:/Users/CURRENT-WORKING-DIRECTORY/SAVE_PATH_INITIAL_TRAINING/chkpts/model.ckpt" # Change this directory to wherever you want to save the training checkpoints.
	save_path_lowest = "C:/Users/CURRENT-WORKING-DIRECTORY/SAVE_PATH_INITIAL_TRAINING/best_chkpt/model.ckpt" # This is the path where the parameters, resulting in the lowest loss yet, will be saved.

	batch_gen = BatchLoader(batch_size = 2048,
			minibatch_size = minibatch_size,
			num_data_points = 22528,
			cwd = "./",
			image_folder_relative = "IMAGE_DATA_2",
			ew_file = "_elbow_wrist.txt",
			wh_file = "_wrist_hand.txt") # Batch generator initialization.

	if mode == "TRAIN":
		minibatches_processed = 0 # For tensorboard logging.
		loss_val = 0
		lowest_loss_yet = 0.00059
		with tf.Session() as sess:
			if not restore_saved_model:
				sess.run(init)
				print("Model initialized from scratch.")
			else:
				saver.restore(sess, save_path) # _lowest) # Loads from lowest loss variable set.
				print("Model restored from checkpoint.")


			# log_all_var() # For tensorboard logging.

			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter('logs/' + str(log_folder) + '/train',
                                      sess.graph)

			printed_once = False
			batch_ctr = 0
			X_buffer, y_buffer = batch_gen.load_entire_train_set() #<--   
			while batch_gen.epochs_passed != num_epochs:            # |
				# X_buffer, y_buffer = batch_gen.get_next_batch()   # |
				m_batch = X_buffer.shape[0]
				num_slices = m_batch // minibatch_size
				batch_loss = 0
				epoch_loss = 0
				tick = time.time()
				for i in range(num_slices): # *IMPORTANT* - THIS WILL NEGLECT THE LAST 'NUM_DATA_POINT % MINIBATCH_SIZE' EXAMPLES.
					start = i * minibatch_size
					end = start + minibatch_size
					X_mb = X_buffer[start:end]
					y_mb = y_buffer[start:end]
					_, loss_val = sess.run([optimizer, loss], feed_dict = {X:X_mb, y:y_mb})
					batch_loss += (loss_val / 256)
					epoch_loss += (loss_val / num_slices)
					minibatches_processed += 1 # For tensorboard logging.
					if i % 32 == 0:
						summary = sess.run(merged, feed_dict = {X:X_mb, y:y_mb})
						train_writer.add_summary(summary, minibatches_processed)
					if i % 256 == 0:
						print("Epochs passed: " + str(batch_gen.epochs_passed) + " | Batch processed: " + str(i / 256) + " | Loss: " + str(batch_loss))
						if batch_loss < lowest_loss_yet:
							lowest_loss_yet = batch_loss
							tick__for_saver = time.time()
							_save_path = saver.save(sess, save_path_lowest)
							tock_for_saver = time.time()
							print("\nBEST MODEL YET!\n")
							print("Saved model variables to path '{}' in {:.3f} seconds.".format(_save_path, tock_for_saver - tick__for_saver))
						batch_loss = 0

				tock = time.time()

				batch_gen.epochs_passed += 1 # Incrementing batch manually.

				# Code for printing out the correct zero-indexed batch number just processed. Useless in this script.
				if batch_gen.current_batch != 0:
					batch_ctr = batch_gen.current_batch - 1
				else:
					batch_ctr = 10
				# --------------------------------------------------------------------------------------------------
				
				print("Epochs passed: " + str(batch_gen.epochs_passed) + " | Epoch time: " + str(tock - tick) + " sec." + " | Epoch loss: " + str(batch_loss))

				if (batch_gen.epochs_passed % save_every_n_epochs == 0):
					tick__for_saver = time.time()
					_save_path = saver.save(sess, save_path)
					tock_for_saver = time.time()
					print("Saved model variables to path '{}' in {:.3f} seconds.".format(_save_path, tock_for_saver - tick__for_saver))
				
	train_writer.close()	
				
	if mode == "VALIDATE":
		loss_val = 0
		validation_batch_loss = 0
		with tf.Session() as sess:
			# sess.run(init)
			# print("Model initialized from scratch.")
			saver.restore(sess, save_path) # , save_path_lowest)
			print("Model restored from checkpoint.")
			writer = tf.summary.FileWriter('logs', sess.graph) # For Tensorboard
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
	
			writer.close() # For Tensorboard

def main():
	model(minibatch_size = 16, num_epochs = 500, learning_rate = 0.0001, mode = "TRAIN", restore_saved_model = True, log_folder = '1') # Uncomment this for training, while commenting out the line below.
	# model(minibatch_size = 16, num_epochs = 8, learning_rate = 0.001, mode = "VALIDATE") # Uncomment this for validation, while commenting out the line above.

if __name__ == '__main__':
	main()