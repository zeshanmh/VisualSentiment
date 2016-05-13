import numpy as np
import sys
import cv2
import os

from data_util import *

NUM_PIXELS = 512*512
NUM_HIST = 512

def pixel_extractor(basepath, img_names): 
	# basepath = '../../data/groupdataset_release/resize_images'
	num_images = len(img_names)
	X = np.zeros((num_images, 3*NUM_PIXELS))
	for i,img in enumerate(img_names): 
		img_arr = cv2.imread(os.path.join(basepath, 'resize_'+img))
		new_arr = np.zeros((3*NUM_PIXELS,))
		num_elems = img_arr.shape[0]*img_arr.shape[1]*img_arr.shape[2]
		flattened_arr = img_arr.flatten()
		new_arr[:num_elems] = flattened_arr
		X[i,:] = new_arr 
	
	return X 

def color_histogram(basepath, img_names): 
	num_images = len(img_names)
	X = np.zeros((num_images, NUM_HIST))
	for i,img in enumerate(img_names): 
		img_arr = cv2.imread(os.path.join(basepath, img))
		hist = cv2.calcHist([img_arr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = hist.flatten()
		X[i,:] = hist

	return X

if __name__ == '__main__':
    img_names = get_filename_list('../../data/groupdataset_release/file_names.txt')
    color_histogram('../../data/groupdataset_release/images', img_names)
    