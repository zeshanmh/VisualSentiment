import numpy as np
import sys
import cv2
import os
import scipy.io as io

from data_util import *
from EmotionExtractor import EmotionExtractor

NUM_PIXELS = 512*512
NUM_HIST = 512
NUM_IMAGES = 597
BB_CAP = 15
NUM_ATTRIBUTES = 10
NUM_BINS_PER_SIDE = 4
NUM_IMG_FACE_FEATURES = NUM_BINS_PER_SIDE*NUM_BINS_PER_SIDE*2


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


def get_bbs(basepath, img_name): 	
	matfile = img_name[:-4] + '_labels.mat'
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

	return saved_bbs

# not used anymore
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

def get_all_face_features(images_path, faces_folder, classifier):
	img_names = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f)) and 'DS_Store' not in f]
	n_images = len(img_names)
	X = np.zeros((n_images, NUM_IMG_FACE_FEATURES))

	for i, img_name in enumerate(img_names):
		image_path = os.path.join(images_path, img_name)
		face_folder = os.path.join(faces_folder, img_name[:-4])
		X[i,:] = get_image_face_features(image_path, face_folder, classifier)

	return X

def get_image_face_features(img_path, face_folder, classifier):
	image = cv2.imread(img_path)
	face_bbs = np.load(os.path.join(face_folder, 'face_bbs.npy'))
	n_faces = face_bbs.shape[0]
	faces = [face_name for face_name in os.listdir(face_folder) if 'faces' not in face_name and 'DS_Store' not in face_name]
	y = image.shape[0]
	x = image.shape[1]
	
	h = y / NUM_BINS_PER_SIDE
	w = x / NUM_BINS_PER_SIDE

	emotion_extractor = EmotionExtractor()
	feature_vec = np.zeros(NUM_IMG_FACE_FEATURES)

	print face_folder

	for k in xrange(n_faces):
		face_path = os.path.join(face_folder, faces[k])
		face = cv2.imread(face_path)
		face_bb = face_bbs[k,:]
		xf, yf, wf, hf = [int(c) for c in face_bb]
		x_center = xf + (wf/2)
		y_center = yf + (hf/2)
		i = y_center / h
		j = x_center / w
		bin_num = i * NUM_BINS_PER_SIDE + j

		emotion_extractor.set_face(face)
		face_features = emotion_extractor.extract_emotion_features()
		prediction = classifier.predict_single_face(face_features)
		feature_vec[2*bin_num + prediction] += 1

	return feature_vec

def get_image_pose_features(img_path, orientations):
	# TODO
	pass

def get_image_group_features(img_path, groups):
	# TODO
	pass



	# for i in xrange(0, image.shape[0]-h, h): 
	# 	for j in xrange(0, image.shape[1]-w, w): 
	# 		bin_num = (i / h) * N_BINS_PER_SIDE + (j / w)
	# 		region_proposal = image[i:i+h,j:j+w]

	# 		for k,face_bb in face_bbs: 
	# 			xf, yf, wf, hf = face_bb 
	# 			x_center = xf + (wf/2)
	# 			y_center = yf + (hf/2)

	# 			if x_center => j and x_center < j + w \
	# 				and y_center => i and y_center < i + h: 
	# 				prediction = classifier.predict(faces[k])

			


	# test_path = ''
	# pathnames = os.listdir(test_path)

	# X = np.zeros((len(pathnames),))
	# for i,path in pathnames: 
	# 	faces_list, smile_features, predictions, scores = classifier.predict(path)
	# 	ratio_smile = np.sum(predictions) / float(predicitons.shape[0])
	# 	X[i] = ratio_smile 

	# return X 


if __name__ == '__main__':
    img_names = get_filename_list('../../data/groupdataset_release/file_names.txt')
    bb_extractor('../../data/groupdataset_release/annotations/all', img_names)
    