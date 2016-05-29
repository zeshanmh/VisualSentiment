import sys
import os
import cv2
import sklearn
# import analysis
import numpy as np
sys.path.insert(0, './util')
import image_util

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
	extract_faces = True 
	if extract_faces: 
		src_path = '../data/GENKI-R2009a/Subsets/GENKI-4K/files'
		dest_path = './cache/GENKI_faces'
		image_util.extract_GENKI_faces(src_path, dest_path)

	# img_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images.txt"
	# labels_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels.txt"

	# svm = train_smile_extractor(img_path, labels_path)
	# img_path2 = '../data/groupdataset_release/images/466491971_b3bfbce419_o.jpg'
	# faces_list, smile_features, predictions = predict_smiles(img_path2, svm)
	# print faces_list
	# print predictions


# Given an image path and a classifier (svm), this method returns a list of 
# image coordinates of faces, a matrix of smile features and the classifier's
# predictions for each face.
def predict_smiles(img_path, classifier):
	face_extractor = FaceExtractor()
	faces_list, im = face_extractor.detect_faces(img_path)
	faces = face_extractor.get_scaled_faces(faces_list, im)
	n_faces = len(faces)

	emotion_extractor = EmotionExtractor()
	smile_features = np.zeros((n_faces, emotion_extractor.NUM_FEATURES))
	print 'Extracting smiles of faces...'
	for i, face in enumerate(faces):
		emotion_extractor.set_face(face)
		feature_vec = emotion_extractor.extract_smile_features()
		smile_features[i,:] = feature_vec

	print 'Predicting...'
	print smile_features.shape
	predictions = classifier.predict(smile_features)

	return faces_list, smile_features, predictions


def train_smile_extractor(img_path, labels_path): 
	#read in stuff 
	img_base_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/files"
	images_file = open(img_path, 'r')
	filenames = []
	
	filenames = images_file.readlines()
	filenames = [os.path.join(img_base_path, filename.strip()) for filename in filenames]

	labels_file = open(labels_path, 'r')
	lines = labels_file.readlines()
	labels = []
	for line in lines: 
		labels.append(int(line.split()[0]))

	#setup 
	if not os.path.isfile('./cache/emotion_features.npy'):
		print "Extracting emotions..."
		emotion_extractor = EmotionExtractor()

		X = np.zeros((len(filenames), emotion_extractor.NUM_FEATURES))
		y = np.array(labels)


		for i, img_name in enumerate(filenames): 
			face_image = cv2.imread(img_name)
			emotion_extractor.set_face(face_image)
			X[i,:] = emotion_extractor.extract_smile_features()

		np.save('./cache/emotion_features', X)
		np.save('./cache/emotion_labels', y)
	else: 
		print "Loading emotions..."
		X = np.load('./cache/emotion_features.npy')
		y = np.load('./cache/emotion_labels.npy')
	print X.shape
	run_again = True
	if not os.path.isfile('./cache/smile_pca.npy') or run_again : 
		print "Running PCA..."
		pca = PCA()
		X_new = pca.fit_transform(X)
		print X_new.shape
		print pca.explained_variance_ratio_[:2000]
		print sum(pca.explained_variance_ratio_[:2000])
		np.save('./cache/smile_pca', X_new)
	else: 
		print "Loading reduced matrix..."
		X_new = np.load('./cache/smile_pca.npy')
	# print X_new.shape
	X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X_new, y, test_size=0.2)

	# run SVM
	svm_model = svm.SVC(kernel="linear", decision_function_shape='ovr')
	print "Fitting..."
	print X_train.shape
	print y_train.shape
	svm_model.fit(X_train, y_train)
	print "Predicting..."
	y_predict_train = svm_model.predict(X_train)
	y_predict = svm_model.predict(X_test)

	_, training_error = analysis.output_error(y_predict_train, y_train)
	_, testing_error = analysis.output_error(y_predict, y_test)

	print "training error:", training_error
	print "testing error:", testing_error 

	return svm_model



if __name__ == '__main__':
	main()
