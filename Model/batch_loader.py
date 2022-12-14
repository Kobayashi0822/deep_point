import numpy as np
import os
import cv2
# import progressbar
import time

bl = [] # change.
image_folder_relative_path = "/home/sobits/catkin_ws/src/deep-point/dataset/left/images_l/"
ew_file_path = "/home/sobits/catkin_ws/src/deep-point/dataset/left/elbow_l.txt"
wh_file_path = "/home/sobits/catkin_ws/src/deep-point/dataset/left/wrist_l.txt"
def file_num(file_name):
		return int(file_name[0:-4])

class BatchLoader(object):
	def __init__(self, batch_size = 4096, num_data_points = 24084, cwd = "./", image_folder_relative=image_folder_relative_path , ew_file = ew_file_path, wh_file = wh_file_path, total_num_data_points = 24084, minibatch_size = 32):
		self.l = os.listdir(image_folder_relative)
		# print(len(self.l))
		# print(self.l[0])
		# print(num_data_points)
		# print(len(self.l), num_data_points) # checking.
		print("in BatchLoader",num_data_points)
		print(len(self.l))
		assert(len(self.l) == num_data_points) # UNCOMMENT AND CORRECT THIS LINE.
		# exit()
		# self.l.sort(key = file_num)
		self.ew_full = np.loadtxt(ew_file).reshape(10000,3)
		print("self.ew_full",self.ew_full.shape)
		# self.ew_full=self.ew_full
		# print("self.ew_full",self.ew_full.shape)
		# print(self.ew_full[0:5,0:3])
		self.wh_full = np.loadtxt(wh_file)
		# print("::",self.ew_full.shape)
		# assert(self.ew_full.shape[0] == num_data_points) # UNCOMMENT AND CORRECT THIS LINE.
		# assert(self.wh_full.shape[0] == num_data_points) # UNCOMMENT AND CORRECT THIS LINE.
		# assert(self.ew_full[3][1] != self.wh_full[3][1])
		self.current_batch = 0
		self.batch_size = batch_size
		self.minibatch_size = minibatch_size
		self.total_num_data_points = total_num_data_points
		self.num_data_points = num_data_points
		self.num_full_batches = self.num_data_points // self.batch_size
		self.img_directory = image_folder_relative 
		self.img_h = 480
		self.img_w = 640
		# self.img_h = 170
		# self.img_w = 213

		self.epochs_passed = 0
		# self.num_loaded_images = 0
		# self.X_buffer = None
		# self.y_buffer = None
		# self.current_minibatch = 0

	def get_batch_limits(self):
		start = self.current_batch * self.batch_size
		end = start + self.batch_size
		if float(self.num_full_batches) != (self.num_data_points / self.batch_size):
			if self.current_batch == self.num_full_batches:
				end = self.num_data_points
			self.current_batch += 1
			if self.current_batch > self.num_full_batches:
				self.epochs_passed += 1
				self.current_batch = 0
		else:
			self.current_batch += 1
			if self.current_batch == self.num_full_batches:
				self.epochs_passed += 1
				self.current_batch = 0
		return start, end

	# def load_next_batch():
	# 	global bl # change.
	# 	start, end = self.get_batch_limits()
	# 	m = end - start
	# 	X = np.empty(shape = (m, 341, 426, 1))
	# 	assert((341, 426) == (self.img_h, self.img_w)) # Change (341, 426) according to image shape.
	# 	for i in range(start, end):
	# 		if i == start:
	# 			bl.append(self.l[start:end])
	# 		img_path = self.img_directory + self.l[i]
	# 		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	# 		assert(img.shape == (self.img_h, self.img_w)) # Change (341, 426) according to image shape.
	# 		index = i - start
	# 		X[index, :, :, 0] = img
	# 	y = self.ew_full[start:end, 1:7]

	# 	self.X_buffer = X
	# 	self.y_buffer = y
	# 	assert(self.y_buffer.shape[0] == self.X_buffer.shape[0])
	# 	self.num_loaded_images = self.y_buffer.shape[0]

	# def get_minibatch_limits():
	# 	if self.num_loaded_images == self.batch_size:


	# def get_next_minibatch():
	# 	if num_loaded_images == self.batch_size:



	def get_next_batch(self):
		global bl # change.
		start, end = self.get_batch_limits()
		m = end - start
		X = np.empty(shape = (m, self.img_h, self.img_w, 3)) # Change last index to match number of input channels. 1 for grayscale. 3 for color.
		# assert((170, 213) == (self.img_h, self.img_w)) # Change (341, 426) according to image shape. <-- Image shape is now {(170, 213)<--[NumPy array shape]}
		for i in range(start, end):
			# if i == start:
			# 	bl.append(self.l[start:end])
			img_path = self.img_directory + self.l[i]
			img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Change last argument to match whether the image is color or grayscale.
			# assert(img.shape == (self.img_h, self.img_w, 3)) # Change (341, 426) according to image shape. <-- Image shape is now {(170, 213)<--[NumPy array shape]}
			index = i - start
			X[index, :, :, :] = img
		y = self.ew_full[start:end, 1:7]
		# print("get_next_batch")
		return X, y # , bl

	def get_validation_batch(self, remove_stray = False):
		"""
		Input:
		remove_stray, boolean - flag that instructs the function to output the highest possible
								number of images that are a multiple of minibatch (not batch) size
								and neglect the remaining (stray) images at the end.
								(THIS FUNCTIONALITY ***HASN'T BEEN IMPLEMENTED*** YET.)
		Output:
		X, y - validation batch as numpy arrays of appropriate dimensions.
		"""
		global bl
		start = 0
		end = 256

		m = end - start
		print("m",m)
		if remove_stray:
			end = end - m % self.minibatch_size
			m = end - start
			print("ifm",m)
		X = np.empty(shape = (m, self.img_h, self.img_w, 3)) # Change last index to match number of input channels. 1 for grayscale. 3 for color.
		print("X",X.shape)
		# assert((170, 213) == (self.img_h, self.img_w)) # Change (341, 426) according to image shape. <-- Image shape is now {(170, 213)<--[NumPy array shape]}
		# assert(len(self.l) == 312)
		for i in range(start, end):
			# if i == start:
			# 	bl.append(self.l[start:end])
			img_path = self.img_directory + self.l[i]
			img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Change last argument to match whether the image is color or grayscale.
			# print(img.shape)
			assert(img.shape == (self.img_h, self.img_w, 3)) # Change (341, 426) according to image shape.
			index = i - start
			X[index, :, :, :] = img
		y = self.ew_full[start:end, 0:3] # CHANGE THIS BACK! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		# print(type(self.ew_full))
		print(y.shape)
		return X, y

	# def load_entire_train_set(self):
	# 	global bl

	# 	start = 0
	# 	end = self.num_data_points
	# 	m = end - start
	# 	m = 20672 # for neglecting bad training data
	# 	X = np.empty(shape = (0, self.img_h, self.img_w, 3)) # last index is 3 for color images. (should be 1 if grayscale)
	# 	y = np.empty(shape = (0, 8))
	# 	assert((170, 213) == (self.img_h, self.img_w)) # Change (341, 426) according to image shape. Now changed to 170, 213
	# 	for i in range(start, end):
	# 		if (i >= 10339 and i < 12042) or (i >= 22375):
	# 			continue
	# 		# if i == start:
	# 		# 	bl.append(self.l[start:end])
	# 		print(i)
	# 		bl.append(self.l[i])
	# 		img_path = self.img_directory + self.l[i]
	# 		img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Change last argument to match whether the image is color or grayscale.
	# 		assert(img.shape == (self.img_h, self.img_w, 3)) 
	# 		# index = i - start
	# 		# X[i, :, :, :] = img
	# 		# y[i, :] = self.ew_full[i]
	# 		X = np.append(X, np.array([img]), axis = 0)
	# 		y = np.append(y, np.array([self.ew_full[i]]), axis = 0)
	# 	# y = self.ew_full[start:end] # , 1:7]
	# 	assert(y.shape[0] % 64 == 0)
	# 	assert(X.shape[0] % 64 == 0)
		
	# 	return X, y

	def load_entire_train_set(self):
		global bl

		start = 0
		# end = self.num_data_points
		end=100
		print("end",end)
		# m = end - start
		m=100
		print("m",m)
		# m = 20672 # for neglecting bad training data
		X = np.empty(shape = (m, self.img_h, self.img_w, 3)) # last index is 3 for color images. (should be 1 if grayscale)
		y = np.empty(shape = (m, 3))
		print(self.img_h,self.img_w)
		# assert((170, 213) == (self.img_h, self.img_w)) # Change (341, 426) according to image shape. Now changed to 170, 213
		index = 0
		for i in range(start, end):
			# index = i
			if (i >= 10339 and i < 12042) or (i >= 22375):
				if i >= 22375:
					break
				else:
					continue
			if i == 12042:
				index = 12042 - 1703

			# if i == start:
			# 	bl.append(self.l[start:end])
			print(i, end = "\r")
			bl.append(self.l[i])
			img_path = self.img_directory + self.l[i]
			img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Change last argument to match whether the image is color or grayscale.
			print(self.img_h,self.img_w,img.shape)
			assert(img.shape == (self.img_h, self.img_w, 3)) 
			# index = i - start
			
			# cv2.imshow("test",img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			print("img",img.shape)
			X[index, :, :, :] = img
			print(index)
			# exit()
			# print(self.ew_full.shape)
			# print(self.ew_full[i, 0:3])
			# print(y.shape)
			# print(y[index, :])
			y[index, :] = self.ew_full[i, 0:3]
			# X = np.append(X, np.array([img]), axis = 0)
			# y = np.append(y, np.array([self.ew_full[i]]), axis = 0)
			index += 1
		# y = self.ew_full[start:end] # , 1:7]
		# assert(y.shape[0] % 64 == 0)
		# assert(X.shape[0] % 64 == 0)
		
		return X, y


	def save_batches(self):
		"""DEPRECATED! DON'T USE THIS FUNCTION"""
		print("Deprecated! Don't use this function.")
		raise BaseException
		"""-----------------------------------"""
		print("Saving batches as .npy files...")
		bar = progressbar.ProgressBar(maxval = self.num_full_batches + 1, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), " | ", progressbar.Timer()])
		bar.start()
		for i in range(self.num_full_batches + 1):
			X, y = self.get_next_batch()
			"""CORRECTNESS ASSERTION"""
			im_name_1 = self.l[3 + (i * 4096)]
			im_name_2 = self.l[7 + (i * 4096)]
			im_num_1 = float(file_num(im_name_1))
			im_num_2 = float(file_num(im_name_2))
			assert(y[3 + (i * 4096), 0] == im_num_1)
			assert(y[7 + (i * 4096), 0] == im_num_2)
			"""---------------------"""
			# np.save("X_" + str(i), X)
			# np.save("y_" + str(i), X)
			bar.update(i + 1)
		bar.finish()

def main():
	global bl

	batches = BatchLoader(batch_size = 2048,
		minibatch_size = 8,
		num_data_points = 10000, #modified for testing, originally 24084
		cwd = "./",
		image_folder_relative = image_folder_relative_path,
		ew_file = ew_file_path,
		wh_file = wh_file_path)
	# batches.save_batches()
	#################UNCOMMENT THIS TO CHECK###########################
	# tick = time.time()
	# X0, y0, bl = batches.get_next_batch()
	# tock = time.time()
	# print("After 1st Batch...")
	# print("bl_size: ", len(bl))
	# print(bl)

	# vtr = 0
	# rouge_list = []
	# for i in range(len(bl[0])):
	# 	assert(float(file_num(bl[0][i])) == y0[i][0])

	# print("rouge_list: ", rouge_list)
	
	# tick2 = time.time()
	# X1, y1, bl = batches.get_next_batch()
	# tock2 = time.time()
	# for i in range(len(bl[0]), len(bl[1])):
	# 	assert(float(file_num(bl[1][i - len(bl[0])])) == y0[i][0])

	# print("After 2nd Batch...")
	# print("bl_size: ", len(bl))
	# # print(bl)

	# print("Time taken for {} batch retrieval = {} ms. ({} sec.)".format("1st", str((tock - tick) * 1000), str(tock - tick)))
	# print("Time taken for {} batch retrieval = {} ms. ({} sec.)".format("2nd", str((tock2 - tick2) * 1000), str(tock2 - tick2)))
	#########################################################################

	# Testing the validation_batch_loading function. ##############################
	# X_val, y_val = batches.get_validation_batch(True)
	# print(X_val.shape, y_val.shape)
	# assert(X_val.shape[0] == y_val.shape[0])
	# print("Checking validation batch validity!")
	# for i in range(batches.num_data_points, batches.total_num_data_points - ((batches.total_num_data_points - batches.num_data_points) % batches.minibatch_size)):
	# 	print(float(file_num(batches.l[i])), y_val[i - batches.num_data_points][0])
	# 	assert(float(file_num(batches.l[i])) == y_val[i - batches.num_data_points][0])
	########################################################################

# ???????????????????
	# batches = BatchLoader(batch_size = 16,
	# 	minibatch_size = 8,
	# 	num_data_points = 256, #modified for testing, originally 24084
	# 	cwd = "./",
	# 	image_folder_relative = image_folder_relative_path,
	# 	ew_file = ew_file_path,
	# 	wh_file = wh_file_path,
	# 	total_num_data_points = 312)

	X, y = batches.get_validation_batch()
	print("X",X.shape[0])
	print("y",y.shape[0])
	assert(X.shape[0] == y.shape[0])
	assert(X.shape[0] == 256)
	# print(len(bl[0]))
	# for i in range(X.shape[0]):
	# 	f_num = file_num(bl[0][i])
	# 	assert(float(f_num) == y[i][0])

if __name__ == '__main__':
	main()