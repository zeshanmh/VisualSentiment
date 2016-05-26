import sys
import os
import cv2
import sklearn
import numpy as np
sys.path.insert(0, './util')

from FaceExtractor import FaceExtractor
from EmotionExtractor import EmotionExtractor
from sklearn import svm
from sklearn.decomposition import PCA


def main():
	## Face Extraction ## 
	# img_path = "../data/groupdataset_release/images/Library3.jpg"
	# face_extractor = FaceExtractor()
	# faces_lists, image = face_extractor.detect_faces(img_path)
	# for face_list in faces_lists: 
	# 	for (x,y,w,h) in face_list: 

	img_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt"
	labels_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt"
	train_smile_extractor(img_path, labels_path)
	

def train_smile_extractor(img_path, labels_path): 
	#read in stuff 
	img_base_path = "../data/GENKI-R2009a/Subsets/GENKI-4K"
	images_file = open(img_path, 'r')
	filenames = []
	for file in images_file.readline(): 
		filenames.append(os.path.join(img_base_path, file.strip()))

	labels_file = open(labels_path, 'r')
	lines = labels_file.readlines()
	labels = []
	for line in lines: 
		labels.append(int(line.split()[0]))

	#setup 
	emotion_extractor = EmotionExtractor()
	n_cells = emotion_extractor.NORMALIZED_SIZE
	n_gaussians = emotion_extractor.NUM_GAUSSIANS
	X = np.zeros((len(filenames), n_cells*n_cells*n_gaussians*2))
	y = np.array(labels)

	for i, img_name in enumerate(filenames): 
		face_image = cv2.imread(img_name)
		emotion_extractor.set_face(face_image)
		X[i,:] = emotion_extractor.extract_smile_features()

	pca = PCA()
	pca.fit(X)

	X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2)







if __name__ == '__main__':
	main()
