import numpy as np
import cv2
import argparse
import os
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mirror", action = "store_true", default = False, help = "Flag that insturcts the augmentation script to flip images and the corresponding skeleton data's X coordinates.")
parser.add_argument("-v", "--verbose", action = "store_true", default = False, help = "Flag that insturcts the augmentation script show a progress bar while performing augmentation.")
parser.add_argument("-c", "--check", action = "store_true", default = False, help = "Flag that insturcts the augmentation script to check one on one correspondence between image and skeleton data.")
args = parser.parse_args()

assert(not (args.mirror and args.check))

ew_source = "D:/PRE_PROCESSED_DATA/_elbow_wrist.txt"
wh_source = "D:/PRE_PROCESSED_DATA/_wrist_hand.txt"
ew_dest = "D:/PRE_PROCESSED_DATA/MIRROR_TEST"
wh_dest = "D:/PRE_PROCESSED_DATA/MIRROR_TEST"

if args.check:
	ew_source = "D:/PRE_PROCESSED_DATA/MIRROR_TEST/_elbow_wrist.txt"
	wh_source = "D:/PRE_PROCESSED_DATA/MIRROR_TEST/_wrist_hand.txt"

ew_num = np.loadtxt(ew_source,  dtype = int, usecols = (0,))
wh_num = np.loadtxt(wh_source,  dtype = int, usecols = (0,))
ew_full = np.loadtxt(ew_source)
wh_full = np.loadtxt(wh_source)

wh_max = wh_num.shape[0]
ew_max = ew_num.shape[0]
assert(wh_max == ew_max)

img_source = img_dest = "D:/PRE_PROCESSED_DATA/MIRROR_TEST/IMAGE_DATA_2/"

l = os.listdir("D:/PRE_PROCESSED_DATA/MIRROR_TEST/IMAGE_DATA_2/")
assert(len(l) == 24084)

def file_num(file_name):
    return int(file_name[0:-4])

def last_image_num():
	global ew_source
	global wh_source
	global ew_dest
	global wh_dest
	global ew_num
	global wh_num
	global ew_full
	global wh_full
	global wh_max
	global ew_max
	global l

	l.sort(key = file_num)
	last_file_name = l[-1]
	last_img_num = int(last_file_name[0:-4])
	assert(ew_num[-1] == last_img_num)
	return ew_num[-1]

LAST_IMG_NUM = LAST_IMG_NUM_ORIG = last_image_num()
INDEX_OFFSET = wh_max

def horizontal_flip():
	global ew_source
	global wh_source
	global ew_dest
	global wh_dest
	global ew_num
	global wh_num
	global ew_full
	global wh_full
	global wh_max
	global ew_max
	global LAST_IMG_NUM
	global INDEX_OFFSET
	global LAST_IMG_NUM_ORIG

	if args.verbose:
		bar = progressbar.ProgressBar(maxval = wh_max, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), " | ", progressbar.Timer()])
		bar.start()

	for i in range(ew_max):
		img = cv2.imread(img_source + str(ew_num[i]) + ".PNG", cv2.IMREAD_GRAYSCALE)
		mirror_img = cv2.flip(img, 1)
		MIRROR_IMG_NUM = LAST_IMG_NUM + 1
		img_skeleton_data_ew = ew_full[[i], :]
		ew_orig = ew_full[[i], :]
		img_skeleton_data_wh = wh_full[[i], :]
		wh_orig = wh_full[[i], :]

		img_skeleton_data_ew[0][1] = -img_skeleton_data_ew[0][1]
		img_skeleton_data_wh[0][1] = -img_skeleton_data_wh[0][1]
		
		img_skeleton_data_ew[0][4] = -img_skeleton_data_ew[0][4]
		img_skeleton_data_wh[0][4] = -img_skeleton_data_wh[0][4]

		img_skeleton_data_ew[0][7] = -img_skeleton_data_ew[0][7]
		img_skeleton_data_wh[0][7] = -img_skeleton_data_wh[0][7]

		img_skeleton_data_ew[0][0] = MIRROR_IMG_NUM
		img_skeleton_data_wh[0][0] = MIRROR_IMG_NUM

		assert((ew_orig + img_skeleton_data_ew)[0][1] == 0)
		assert((wh_orig + img_skeleton_data_wh)[0][1] == 0)
		assert((ew_orig + img_skeleton_data_ew)[0][4] == 0)
		assert((wh_orig + img_skeleton_data_wh)[0][4] == 0)

		ew_full = np.append(arr = ew_full, values = img_skeleton_data_ew, axis = 0)
		wh_full = np.append(arr = wh_full, values = img_skeleton_data_wh, axis = 0)

		cv2.imwrite(img_dest + str(MIRROR_IMG_NUM) + ".PNG", mirror_img)
		LAST_IMG_NUM += 1

		if args.verbose:
			bar.update(i + 1)

	"""VALIDITY ASSERTION"""
	max_iter = ew_full.shape[0]
	offset = max_iter / 2 
	for i in range(offset):
		assert(ew_full[i][1] + ew_full[i + offset][1] == 0)
		assert(wh_full[i][1] + wh_full[i + offset][1] == 0)
		assert(ew_full[i][4] + ew_full[i + offset][4] == 0)
		assert(wh_full[i][4] + wh_full[i + offset][4] == 0)
		assert(ew_full[i][7] + ew_full[i + offset][7] == 0)
		assert(wh_full[i][7] + wh_full[i + offset][7] == 0)
		if i == (offset - 1):
			print(ew_full[i + offset][0], ew_full[i + offset][1])
			assert(ew_full[i + offset][0] == float(LAST_IMG_NUM_ORIG + offset))
			assert(wh_full[i + offset][0] == float(LAST_IMG_NUM_ORIG + offset))
	"""-------------------"""

	ew_filename = ew_dest + "/_elbow_wrist.txt"
	wh_filename = wh_dest + "/_wrist_hand.txt"
	np.savetxt(ew_filename, ew_full)
	np.savetxt(wh_filename, wh_full)

	if args.verbose:
		bar.finish()

def assert_one_on_one_correspondence():
	print("BEGIN!")
	global ew_source
	global wh_source
	global ew_dest
	global wh_dest
	global ew_num
	global wh_num
	global ew_full
	global wh_full
	global wh_max
	global ew_max
	global LAST_IMG_NUM
	global INDEX_OFFSET
	global LAST_IMG_NUM_ORIG
	global l

	wh_max = 24084
	ew_max = ew_full.shape[0]
	assert(wh_max == ew_max)

	if args.verbose:
		bar = progressbar.ProgressBar(maxval = wh_max, widgets = ["Checking data correspondence: ", progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), " | ", progressbar.Timer()])
		bar.start()

	i = 0
	li = 0
	assert(len(l) == ew_max)
	assert(len(l) == wh_max) 
	while li < len(l) and i < ew_max:
		assert(li == i)
		if (li == (len(l) - 1)):
			assert(li == i)
		i_file_name = l[li]
		if li == 0:
			assert(i_file_name == "1.PNG")
		i_num = file_num(i_file_name)
		
		assert(float(i_num) == ew_full[i][0])
		assert(float(i_num) == wh_full[i][0])

		if args.verbose:
			bar.update(i + 1)
		i += 1
		li += 1

	assert(li == i)
	assert(li == 24084 and i == 24084)

	if args.verbose:
		bar.finish()

	print("END!")

def main():
	if args.mirror:
		horizontal_flip()
	if args.check:
		assert_one_on_one_correspondence()


if __name__ == '__main__':
	main()