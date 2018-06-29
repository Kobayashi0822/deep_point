"""
This script automates the process of pre-processing, resizing, and cleaning of the captured data points.
"""

import numpy as np
import cv2
import progressbar

ew_source = "C:/Users/praty/Desktop/DL/IIIT-A Research Internship/HGR/Code/Data_Collection/_elbow_wrist.txt"
wh_source = "C:/Users/praty/Desktop/DL/IIIT-A Research Internship/HGR/Code/Data_Collection/_wrist_hand.txt"
ew_dest = "D:/PRE_PROCESSED_DATA"
wh_dest = "D:/PRE_PROCESSED_DATA"

ew_num = np.loadtxt(ew_source,  dtype = int, usecols = (0,))
wh_num = np.loadtxt(wh_source,  dtype = int, usecols = (0,))
ew_full = np.loadtxt(ew_source)
wh_full = np.loadtxt(wh_source)

wh_max = wh_num.shape[0]
ew_max = ew_num.shape[0]

img_dest = "D:/PRE_PROCESSED_DATA/IMAGE_DATA/"


def resolve_skeleton_data():
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

	for i in range(wh_max):
		if wh_num[i] != ew_num[i]:
			ew_num = np.delete(ew_num, obj = i, axis = 0)
			ew_full = np.delete(ew_full, obj = i, axis = 0)

	wh_max = wh_num.shape[0]
	ew_max = ew_num.shape[0]

def valid():
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

	l_ew = []
	l_wh = []
	ew_offset = 0
	for i in range(wh_max):
		if wh_full[i, 0] != ew_full[i + ew_offset, 0]:
			l_ew.append((i + ew_offset, ew_full[i + ew_offset]))
			l_wh.append((i, wh_full[i]))
			ew_offset += 1
	return not bool(ew_offset)

def save_skeleton_data():
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

	assert(wh_max == ew_max)
	
	ew_filename = ew_dest + "/_elbow_wrist.txt"
	wh_filename = wh_dest + "/_wrist_hand.txt"
	np.savetxt(ew_filename, ew_full)
	np.savetxt(wh_filename, wh_full)

def process_images(verbose = True):
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

	assert(wh_max == ew_max)
	
	print("PROCESSING IMAGES...")
	if verbose:
		bar = progressbar.ProgressBar(maxval = wh_max, widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

	for i in range(ew_max):
		img_num = wh_num[i]
		img_path = "C:/Users/praty/Desktop/DL/IIIT-A Research Internship/HGR/Code/Data_Collection/IMAGE_DATA/" + str(img_num) + ".PNG"

		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		scale_factor = 3
		new_dim = (img.shape[1] // scale_factor, img.shape[0] // scale_factor)
		resized_img = cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)
		cv2.imwrite(img_dest + str(img_num) + ".PNG", resized_img)

		#your code here.
		if verbose:
			bar.update(i + 1)
	if verbose:
		bar.finish()


def main():
	resolve_skeleton_data()
	if valid():
		print("SAVING RESOLVED SKELETON DATA...")
		save_skeleton_data()
	process_images()

if __name__ == '__main__':
	main()