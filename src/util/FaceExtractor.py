import numpy as np
import cv2
import scipy.io as io

class FaceExtractor:

	def __init__(self):

		casc_path1 = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
		casc_path2 = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_mcs_lefteye.xml"
		self.face_cascade = cv2.CascadeClassifier(casc_path1)
		self.side_cascade = cv2.CascadeClassifier(casc_path2)

	def detect_faces(self, img_path):
		image = cv2.imread(img_path)

		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		faces = self.face_cascade.detectMultiScale(
			gray_img,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(10, 10),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		faces2 = self.side_cascade.detectMultiScale(
			gray_img,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(10, 10),
			flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)

		print "Found {0} faces!".format(len(faces) + len(faces2))

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		for (x, y, w, h) in faces2:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv2.imshow("Faces found" ,image)
		cv2.waitKey(0)



