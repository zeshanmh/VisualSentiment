import numpy as np
import cv2
import os
import sys
import sklearn
import scipy.io as io
import pandas as pd 
import itertools
sys.path.insert(0, './util')
import analysis

from FaceExtractor import FaceExtractor
from EmotionExtractor import EmotionExtractor
from TorsoExtractor import TorsoExtractor
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.externals import joblib


class EmotionSVM: 

	IMAGE_SIZE = 48
	NUM_EMOTION_IMAGES = 35887
	NORMALIZED_SIZE = 64
	SUBSET_EMOTION_IMAGES = 2000
	emotion_dict = {
		'smile': -1, 
		'anger': 0, 
		'disgust': 1, 
		'fear': 2, 
		'happy': 3, 
		'sad': 4, 
		'surprise': 5, 
		'neutral': 6
	}

	def __init__(self, train_img_path, labels_path, test_img_path, emotion, classifier=None, dump=False, fit=True): 
		self.train_img_path = train_img_path
		self.test_img_path = test_img_path 
		self.labels_path = labels_path
		self.emotion = self.emotion_dict[emotion]
		self.classifier = None
		if not fit: 
			if self.emotion == -1: 
				self.classifier = joblib.load('./svm_models/svm_smile_model.pkl')
			else: 
				self.classifier = joblib.load('./svm_models/svm%s_model.pkl'%str(self.emotion)) 
		self.dump = dump


	def train(self): 
		#read in stuff 
		if self.emotion == -1: 
			X, y = self.get_smile_feature_matrix()
		else: 
			X, y = self.get_feature_matrix()

		# img_base_path = "./cache/GENKI_faces"
		# images_file = open(self.train_img_path, 'r')

		# filenames = []
		# filenames = images_file.readlines()
		# filenames = [os.path.join(img_base_path, filename.strip()) for filename in filenames]

		# labels_file = open(labels_path, 'r')
		# lines = labels_file.readlines()
		# labels = []
		# for line in lines: 
		# 	labels.append(int(line.split()[0]))

		# #setup 
		# run_extraction_again = False
		

		# if not os.path.isfile('./cache/emotion_features.npy') or run_extraction_again:
		# 	print "Extracting emotions..."
		# 	emotion_extractor = EmotionExtractor()

		# 	X = np.zeros((len(filenames), emotion_extractor.NUM_FEATURES))
		# 	y = np.array(labels)


		# 	for i, img_name in enumerate(filenames):
		# 		# check if image is in 
		# 		print "Extracting emotions for image:", str(i)
		# 		face_image = cv2.imread(img_name)
		# 		emotion_extractor.set_face(face_image)
		# 		X[i,:] = emotion_extractor.extract_emotion_features()

		# 	np.save('./cache/emotion_features', X)
		# 	np.save('./cache/emotion_labels', y)
		# else: 
		# 	print "Loading emotions..."
		# 	X = np.load('./cache/emotion_features.npy')
		# 	y = np.load('./cache/emotion_labels.npy')


		# run_PCA_again = True
		# ###############################
		# ###### UNCOMMENT FOR PCA ######
		# ###############################

		# if not os.path.isfile('./cache/smile_pca.npy') or run_PCA_again : 
		# 	print "Running PCA..."
		# 	pca = PCA()
		# 	X_new = pca.fit_transform(X)
		# 	np.save('./cache/smile_pca', X_new)
		# 	X_new = X_new[:,:3000]
		# else: 
		# 	print "Loading reduced matrix..."
		# 	X_new = np.load('./cache/smile_pca.npy')

		print "X shape:", X.shape
		X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2)

		# run SVM
		if self.classifier == None: 
			print "Fitting..."
			svm_model = svm.SVC(kernel='linear', decision_function_shape='ovr', verbose=True)
			svm_model.fit(X_train, y_train)
			self.classifier = svm_model
		else: 
			svm_model = self.classifier

		if self.dump: 
			if self.emotion == -1: 
				joblib.dump(svm_model, './svm_models/svm_smile_model.pkl')
			else: 
				joblib.dump(svm_model, './svm_models/svm%s_model.pkl' % str(self.emotion))

		print "Predicting..."
		y_predict_train = svm_model.predict(X_train)
		y_predict = svm_model.predict(X_test)
		_, training_error = analysis.output_error(y_predict_train, y_train)
		_, testing_error = analysis.output_error(y_predict, y_test)

		print "training error:", training_error
		print "testing error:", testing_error 


	def predict_single_face(self, face_features):
		prediction = self.classifier.predict(face_features.reshape(1, -1))
		return prediction[0]

	def predict(self, image_path):
		face_extractor = FaceExtractor()
		faces_list, im = face_extractor.detect_faces(image_path)
		faces = face_extractor.get_scaled_faces(faces_list, im)
		n_faces = len(faces)

		emotion_extractor = EmotionExtractor()
		smile_features = np.zeros((n_faces, emotion_extractor.NUM_FEATURES))
		print 'Extracting smiles of faces...'
		for i, face in enumerate(faces):
			emotion_extractor.set_face(face)
			feature_vec = emotion_extractor.extract_emotion_features()
			smile_features[i,:] = feature_vec

		###############################
		###### UNCOMMENT FOR PCA ######
		###############################
		# do PCA
		# pca = PCA()
		# smile_features_pcad = pca.fit_transform(smile_features)

		print 'Predicting...'
		predictions = self.classifier.predict(smile_features)
		scores = self.classifier.decision_function(smile_features)

		return faces_list, smile_features, predictions, scores


	def get_smile_feature_matrix(self): 
		img_base_path = "./cache/GENKI_faces"
		images_file = open(self.train_img_path, 'r')

		filenames = []
		filenames = images_file.readlines()
		filenames = [os.path.join(img_base_path, filename.strip()) for filename in filenames]

		labels_file = open(self.labels_path, 'r')
		lines = labels_file.readlines()
		labels = []
		for line in lines: 
			labels.append(int(line.split()[0]))

		#setup 
		run_extraction_again = False
		

		if not os.path.isfile('./cache/emotion_features.npy') or run_extraction_again:
			print "Extracting emotions..."
			emotion_extractor = EmotionExtractor()

			X = np.zeros((len(filenames), emotion_extractor.NUM_FEATURES))
			y = np.array(labels)

			y_emotion_idx = (y == self.emotion)
			np.random.choice(y_emotion_idx, )

			for i, img_name in enumerate(filenames):
				# check if image is in 
				print "Extracting emotions for image:", str(i)
				face_image = cv2.imread(img_name)
				emotion_extractor.set_face(face_image)
				X[i,:] = emotion_extractor.extract_emotion_features()

			np.save('./cache/emotion_features', X)
			np.save('./cache/emotion_labels', y)
		else: 
			print "Loading emotions..."
			X = np.load('./cache/emotion_features.npy')
			y = np.load('./cache/emotion_labels.npy')

		return X, y


	def get_feature_matrix(self): 
		face_pd = pd.read_csv('../data/fer2013/fer2013.csv')
		images = face_pd['pixels'].as_matrix() 

		run_extraction_again = True
		
		if not os.path.isfile('./cache/%s_features.npy'%str(self.emotion)) or run_extraction_again: 

			if not os.path.isfile('./cache/X_im.npy'): 
				X_im = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_EMOTION_IMAGES))
				for i,img in enumerate(images): 
					print "image %d" % i
					# print img.split()[:96]
					img_l = [int(x) for x in img.split()]
					im_arr = np.reshape(np.array(img_l), (self.IMAGE_SIZE, self.IMAGE_SIZE)).astype('uint8')
					# print im_arr[:2,:]
					# im_arr = cv2.resize(im_arr, (self.NORMALIZED_SIZE, self.NORMALIZED_SIZE))
					# print im_arr[:2,:]
					X_im[:,:,i] = im_arr 
				np.save('./cache/X_im', X_im)
			else: 
				X_im = np.load('./cache/X_im.npy').astype('uint8')

			emotion_extractor = EmotionExtractor()
			X = np.zeros((self.NUM_EMOTION_IMAGES, emotion_extractor.NUM_FEATURES))
			y = face_pd['emotion'].as_matrix() 

			#get subset
			if self.emotion != self.emotion_dict['neutral']: 
				y_emotion_idxs = [i for i,x in enumerate(y == self.emotion) if x == 1]
				y_not_idxs = [i for i,x in enumerate(y == self.emotion_dict['neutral']) if x == 1]
			else: 
				y_emotion_idxs = [i for i,x in enumerate(y == self.emotion) if x == 1]
				y_not_idxs = [i for i,x in enumerate(y == self.emotion_dict['happy']) if x == 1]

			y_idxs = np.random.choice(y_emotion_idxs, self.SUBSET_EMOTION_IMAGES, replace=False)
			y_not_idxs = np.random.choice(y_not_idxs, self.SUBSET_EMOTION_IMAGES, replace=False)
			y_chosen_idxs = []
			y_chosen_idxs.extend(y_idxs)
			y_chosen_idxs.extend(y_not_idxs)	
			np.random.shuffle(y_chosen_idxs)
			# print y[y_chosen_idxs[4]]
			# cv2.imshow('face', X_im[:,:,y_chosen_idxs[4]])
			# cv2.waitKey(0)

			y_subset = []
			X_subset = np.zeros((2*self.SUBSET_EMOTION_IMAGES, emotion_extractor.NUM_FEATURES))

			print "Extracting emotions..."
			for i,idx in enumerate(y_chosen_idxs): 
				print "image %d" % i
				if y[idx] == self.emotion: 
					y_subset.append(1)
				else: 
					y_subset.append(0)
				face_image = X_im[:,:,idx]
				emotion_extractor.set_face(face_image)
				X_subset[i,:] = emotion_extractor.extract_emotion_features()
			# y_subset = [y[idx] for idx in y_chosen_idxs]
			

			# idxs = y_chosen_idxs[:10]
			# y_10 = []
			# for idx in idxs: 
			# 	if y[idx] == self.emotion: 
			# 		y_10.append(1)
			# 	else: 
			# 		y_10.append(0)

			# for i,idx in enumerate(idxs): 
			# 	img = X_im[:,:,idx]
			# 	print 'label:', y[idx]
			# 	print 'real label:', y_10[i]
			# 	cv2.imshow('face', img)
			# 	cv2.waitKey(0)


			
			# for i in xrange(2*self.SUBSET_EMOTION_IMAGES): 
			# 	print "image %d" % i
			# 	face_image = X_im[:,:,y_chosen_idxs[i]]
			# 	emotion_extractor.set_face(face_image)
			# 	X_subset[i,:] = emotion_extractor.extract_emotion_features()
			# X_subset = np.load('./cache/%s_features.npy' % str(self.emotion))

			np.save('./cache/%s_features' % str(self.emotion), X_subset)
			np.save('./cache/%s_labels' % str(self.emotion), y_subset)
		else: 
			print "Loading emotions..."
			X_subset = np.load('./cache/%s_features.npy' % str(self.emotion))
			y_subset = np.load('./cache/%s_labels.npy' % str(self.emotion))

		return X_subset, y_subset 

