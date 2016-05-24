import numpy as np
import cv2
import os
import scipy.io as io

from feature_extractors import *

class FaceExtractor:

	def __init__(self):
		dir_path = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/"
		file_names = os.listdir(dir_path)
		self.cascades = []
		for i, filename in enumerate(file_names): 
			pathname = os.path.join(dir_path,filename)
			#or 'ear' in pathname or 'eye' in pathname
			if ('face' in pathname) \
				and 'catface' not in pathname:
				face_cascade = cv2.CascadeClassifier(pathname);
				self.cascades.append(face_cascade)
				

	def detect_faces(self, img_path):
		image = cv2.imread(img_path)
		bbs = get_bbs('../data/groupdataset_release/annotations/all', img_path.split('/')[-1])
		print bbs

		for bb in bbs: 
			x, y, width, height = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
			cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)

		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


		faces_lists = []
		for i,classifier in enumerate(self.cascades):
			face_list = self.cascades[i].detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=3,
				minSize=(10, 10),
				flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			)

			faces_lists.append(list(face_list))

		for face_list in faces_lists: 
			for i in reversed(xrange(len(face_list))): 
				x_face, y_face, w_face, h_face = face_list[i]
				# center_face = (x_face + (w_face / 2), y_face + (h_face / 2))
				valid_face = False 
				for bb in bbs: 
					x, y, w_bb, h_bb = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
					h_middle = h_bb / 2; 

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
		for face_list in faces_lists:
			for (x, y, w, h) in face_list:
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
		cv2.imshow("Faces found" ,image)
		cv2.waitKey(0)



