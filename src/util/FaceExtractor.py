import numpy as np
import cv2
import os
import scipy.io as io

class FaceExtractor:

	def __init__(self):
		dir_path = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/"
		file_names = os.listdir(dir_path)
		self.cascades = []
		# casc_path1 = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
		for i,filename in enumerate(file_names):
			if 'face' in filename:
				face_cascade = cv2.CascadeClassifier(filename);
				self.cascades.append(face_cascade)
		# self.face_cascade = cv2.CascadeClassifier(casc_path1)
		# self.side_cascade = cv2.CascadeClassifier(casc_path2)

		def detect_faces(self, img_path):
			image = cv2.imread(img_path)

		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


		faces_lists = []
		for i,classifier in enumerate(self.cascades):
			face_list = self.face_cascade.detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(10, 10),
				flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			)
			faces_lists.append(face_list)

		print "Found {0} faces!".format(sum([ len(x) for x in faces_lists]))

		# Draw a rectangle around the faces
		for face_list in faces_lists:
			for (x, y, w, h) in face_list:
				cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
				cv2.imshow("Faces found" ,image)
				cv2.waitKey(0)



