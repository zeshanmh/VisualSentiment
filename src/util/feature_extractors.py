import numpy as np
import sys
import cv2
import os
import scipy.io as io

from data_util import *

NUM_PIXELS = 512*512
NUM_HIST = 512
NUM_IMAGES = 597
BB_CAP = 15

def pixel_extractor(basepath, img_names): 
	# basepath = '../../data/groupdataset_release/resize_images'
	X = np.zeros((NUM_IMAGES, 3*NUM_PIXELS))
	for i,img in enumerate(img_names): 
		img_arr = cv2.imread(os.path.join(basepath, 'resize_'+img))
		new_arr = np.zeros((3*NUM_PIXELS,))
		num_elems = img_arr.shape[0]*img_arr.shape[1]*img_arr.shape[2]
		flattened_arr = img_arr.flatten()
		new_arr[:num_elems] = flattened_arr
		X[i,:] = new_arr 
	
	return X 

def color_histogram(basepath, img_names): 
	X = np.zeros((NUM_IMAGES, NUM_HIST))
	for i,img in enumerate(img_names): 
		img_arr = cv2.imread(os.path.join(basepath, img))
		hist = cv2.calcHist([img_arr], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		hist = hist.flatten()
		X[i,:] = hist

	return X

def bb_extractor(basepath, img_names):
	X = np.zeros((NUM_IMAGES, 60))
	for i, fil in enumerate(img_names):
		matfile = fil[:-4] + '_labels.mat'
		# print matfile
		mat_struct = io.loadmat(os.path.join(basepath, matfile))
		hmns = mat_struct['hmns']
		saved_bbs = []
		# count = 0
		for group in xrange(hmns.shape[1]):
			# print group
			for person_list in hmns[0,group]:
				# print len(person_list)
				for person in person_list:
					bb_list = person[3]
					for bb in bb_list:
						saved_bbs.append(bb)

		if len(saved_bbs) > BB_CAP:
			rand_order = np.random.permutation(len(saved_bbs))
			idxs = np.sort(rand_order[:BB_CAP])
			saved_bbs = np.array(saved_bbs)
			saved_bbs = saved_bbs[idxs]
		X[i,:4*len(saved_bbs)] = np.array(saved_bbs).flatten()

	return X

if __name__ == '__main__':
    img_names = get_filename_list('../../data/groupdataset_release/file_names.txt')
    bb_extractor('../../data/groupdataset_release/annotations/all', img_names)
    