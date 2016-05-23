import numpy as np
import cv
import scipy.io as io

class FaceExtractor:

	def __init__(self):

		casc_path = "/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
		self.face_cascade = cv.CascadeClassifier(casc_path)

	def detect_faces(self, img_path):
		image = cv.imread(img_path)

		#changed BGR to RGB. possible error
		gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
		faces = faceCascade.detectMultiScale(
			gray_img,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags = cv.cv.CV_HAAR_SCALE_IMAGE
		)

		print "Found {0} faces!".format(len(faces))

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		cv.imshow("Faces found" ,image)
		cv.waitKey(0)



