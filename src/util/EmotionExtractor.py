import numpy as np
import cv2
import os
import scipy.io as io
import math

from feature_extractors import *

class EmotionExtractor: 

	NORMALIZED_SIZE = 64
	NUM_GAUSSIANS = 5
	DIM_WINDOW = NORMALIZED_SIZE / 4
	NUM_STAGES = 3
	NUM_LEVELS = 3
	NUM_FEATURES = DIM_WINDOW * DIM_WINDOW * NUM_GAUSSIANS * NUM_STAGES * NUM_LEVELS * 2
	
	def __init__(self): 
		self.face = None


	def set_face(self, face_image):
		self.face = face_image

	def build_pyr_level(self, img, sigma=1):
		I1 = cv2.GaussianBlur(img, (3,3), sigma)
		I2 = cv2.GaussianBlur(I1, (3,3), np.sqrt(2)*sigma)
		down = cv2.pyrDown(I2, dstsize=(I2.shape[0]/2, I2.shape[1]/2))
		I_new = cv2.pyrUp(down, dstsize=I2.shape)
		return I1, I2, I_new

	def calc_mean_sd(self, pixel_derivs):
			n_stats = 5
			all_stats = np.zeros(self.DIM_WINDOW*self.DIM_WINDOW*2*n_stats)
			counter = 0
			for i in xrange(0, self.NORMALIZED_SIZE, 4):
				for j in xrange(0, self.NORMALIZED_SIZE, 4):
					# first_window = pixel_derivs[i:i+4,j:j+4,1]
					# print "second window:"
					# print first_window
					flat_window = np.reshape(pixel_derivs[i:i+4,j:j+4,:], (-1, 5))

					for col in xrange(flat_window.shape[1]):
						col_list = list(flat_window[:,col])
						good_list = [x for x in col_list if not math.isnan(x)]
						all_stats[counter] = np.mean(np.array(good_list))
						all_stats[counter+1] = np.std(np.array(good_list))
						counter += 2
	
			return all_stats

	def get_derivs(self, img):
		pixel_derivs = np.zeros((self.NORMALIZED_SIZE, self.NORMALIZED_SIZE, self.NUM_GAUSSIANS))
		for i in xrange(self.NORMALIZED_SIZE):
			for k in xrange(self.NORMALIZED_SIZE): 
				#calculate derivatives 
				deriv_vec = np.zeros((self.NUM_GAUSSIANS,))

				if i != 0 and i != self.NORMALIZED_SIZE - 1: 
					deriv_vec[1] = img[i+1,k] - img[i-1,k]
					deriv_vec[3] = img[i+1,k] - 2 * img[i,k] + img[i-1,k]
				else: 
					deriv_vec[1] = None
					deriv_vec[3] = None

				if k != 0 and k != self.NORMALIZED_SIZE - 1: 
					deriv_vec[0] = img[i,k+1] - img[i,k-1]
					deriv_vec[2] = img[i,k+1] - 2 * img[i,k] + img[i,k-1]					
				else: 
					deriv_vec[0] = None
					deriv_vec[2] = None

				if i != 0 and i != self.NORMALIZED_SIZE - 1 and k != 0 \
					and k != self.NORMALIZED_SIZE - 1: 
					deriv_vec[4] = img[i+1,k+1] - img[i-1,k+1] - img[i+1,k-1] + img[i-1,k-1]
				else: 
					deriv_vec[4] = None

				pixel_derivs[i,k] = deriv_vec

		return pixel_derivs

	def extract_smile_features(self):

		#read in the image 
		gray_img = cv2.cvtColor(self.face, cv2.COLOR_RGB2GRAY)

		#normalize to 64x64; precondition: make sure that face patches are square!!
		gray_img_resize = cv2.resize(gray_img, (self.NORMALIZED_SIZE, self.NORMALIZED_SIZE)) 

		pyramid = []
		img = gray_img_resize

		feature_vec = []

		for s in xrange(self.NUM_STAGES):
			pyr_level = []
			I1, I2, new_img = self.build_pyr_level(img)
			derivs0 = self.get_derivs(img)
			feature_vec.append(self.calc_mean_sd(derivs0))
			
			derivs1 = self.get_derivs(I1)
			feature_vec.append(self.calc_mean_sd(derivs1))
			
			derivs2 = self.get_derivs(I2)
			feature_vec.append(self.calc_mean_sd(derivs2))
			pyramid.append([img, I1, I2])
			img = new_img

		feature_vec = np.array(feature_vec)
		feature_vec = np.reshape(feature_vec, (-1,))

		# print feature_vec[:10	]
		return feature_vec

