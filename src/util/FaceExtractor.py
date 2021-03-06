import numpy as np
import cv2
import os
import scipy.io as io

from feature_extractors import *

class FaceExtractor:

	NORMALIZED_SIZE = 64

	def __init__(self):
		dir_path = "/usr/local/Cellar/opencv3/3.1.0_3/share/OpenCV/haarcascades/"
		file_names = os.listdir(dir_path)
		self.cascades = []

		# ##only use for GENKI
		# filename = 'haarcascade_frontalface_default.xml'
		# pathname = os.path.join(dir_path, filename)
		# face_cascade = cv2.CascadeClassifier(pathname)
		# self.cascades.append(face_cascade)

		#uncomment for regular
		for i, filename in enumerate(file_names): 
			pathname = os.path.join(dir_path,filename)
			#or 'ear' in pathname or 'eye' in pathname

			if ('frontalface' in pathname) or 'profileface' in pathname \
				and 'catface' not in pathname:
				# print pathname
				face_cascade = cv2.CascadeClassifier(pathname);
				self.cascades.append(face_cascade)

	def detect_faces(self, img):
		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		faces_lists = []
		for i,classifier in enumerate(self.cascades):
			face_list = self.cascades[i].detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=3,
				minSize=(20, 20),
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			faces_lists.extend(list(face_list))

		return faces_lists

	def detect_faces_GENKI(self, img_path):
		image = cv2.imread(img_path)

		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		faces_lists = []
		for i,classifier in enumerate(self.cascades):
			face_list = self.cascades[i].detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=3,
				minSize=(20, 20),
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			faces_lists.append(list(face_list))

		for face_list in faces_lists: 
			for i in reversed(xrange(len(face_list))): 
				x_face, y_face, w_face, h_face = face_list[i]
				# center_face = (x_face + (w_face / 2), y_face + (h_face / 2))
				valid_face = False 
				for bb in bbs: 
					x, y, w_bb, h_bb = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
					h_middle = h_bb / 2
					# if center_face[0] > x and center_face[1] > y and center_face[0] < x + w_bb \
					# 	and center_face[1] < y + h_bb: 
					# 	valid_face = True 
					if x_face >= x and x_face + w_face < x + w_bb \
						and y_face >= y and y_face + h_face <= y + h_middle: 
						valid_face = True

				if not valid_face: 
					del face_list[i]				

		print "Found {0} faces!".format(sum([ len(x) for x in faces_lists]))

		# Draw a rectangle around the faces
		# for face_list in faces_lists:
		# 	for (x, y, w, h) in face_list:
		# 		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		# cv2.imshow("Faces found", image)
		# cv2.waitKey(0)

		return faces_lists, image


	# def detect_faces(self, img):
	# 	# img = cv2.imread(img_path)
	# 	# bbs = get_bbs('../data/groupdataset_release/annotations/all', img_path.split('/')[-1])
	# 	# print bbs

	# 	# for bb in bbs: 
	# 	# 	x, y, width, height = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
	# 	# 	cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 2)

	# 	#changed BGR to RGB. possible error
	# 	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


	# 	faces_lists = []
	# 	for i,classifier in enumerate(self.cascades):
	# 		face_list = self.cascades[i].detectMultiScale(
	# 			gray_img,
	# 			scaleFactor=1.1,
	# 			minNeighbors=2,
	# 			minSize=(10, 10),
	# 			flags = cv2.CASCADE_SCALE_IMAGE
	# 		)

	# 		if len(face_list) > 0:
	# 			faces_lists.append(list(face_list))

	# 	# for face_list in faces_lists: 
	# 	# 	for i in reversed(xrange(len(face_list))): 
	# 	# 		x_face, y_face, w_face, h_face = face_list[i]
	# 	# 		# center_face = (x_face + (w_face / 2), y_face + (h_face / 2))
	# 	# 		valid_face = False 
	# 	# 		for bb in bbs: 
	# 	# 			x, y, w_bb, h_bb = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
	# 	# 			h_middle = h_bb / 2
	# 	# 			# if center_face[0] > x and center_face[1] > y and center_face[0] < x + w_bb \
	# 	# 			# 	and center_face[1] < y + h_bb: 
	# 	# 			# 	valid_face = True 
	# 	# 			if x_face >= x and x_face + w_face < x + w_bb \
	# 	# 				and y_face >= y and y_face + h_face <= y + h_middle: 
	# 	# 				valid_face = True

	# 	# 		if not valid_face: 
	# 	# 			del face_list[i]				

	# 	# print "Found {0} faces!".format(sum([ len(x) for x in faces_lists]))

	# 	# Draw a rectangle around the faces
	# 	# for face_list in faces_lists:
	# 	# 	for (x, y, w, h) in face_list:
	# 	# 		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
	# 	# cv2.imshow("Faces found", image)
	# 	# cv2.waitKey(0)
	# 	# print faces_lists
	# 	return faces_lists, img


	def scale_face(self, face_coords, im):
		# print face_coords
		x, y, w, h = face_coords
		face = im[y:y+h,x:x+w]
		# face = im[x:x+w,y:y+h]
		# print face.shape 
		# print type(face[0,0])
		scaled_face = cv2.resize(face, (self.NORMALIZED_SIZE, self.NORMALIZED_SIZE))
		return scaled_face


	def get_scaled_faces(self, face_lists, im):
		scaled_faces = []
		for face_list in face_lists:
			for face in face_list: 
				scaled_faces.append(self.scale_face(face, im))
		return scaled_faces


