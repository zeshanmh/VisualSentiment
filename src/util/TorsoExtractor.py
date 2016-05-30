import numpy as np
import cv2
import os
import scipy.io as io

from feature_extractors import *

class TorsoExtractor: 

	def __init__(self):
		dir_path = '/usr/local/Cellar/opencv/2.4.12_2/share/OpenCV/haarcascades/'
		filename = 'haarcascade_upperbody.xml'
		pathname = os.path.join(dir_path, filename)

		self.cascade = cv2.CascadeClassifier(pathname)

	def detect_torsos(self, img_path):
		image = cv2.imread(img_path)
		bbs = get_bbs('../data/groupdataset_release/annotations/all', img_path.split('/')[-1])
		print bbs

		# for bb in bbs: 
		# 	x, y, width, height = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
		# 	cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), 2)

		#changed BGR to RGB. possible error
		gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		torso_list = self.cascade.detectMultiScale(
				gray_img,
				scaleFactor=1.1,
				minNeighbors=2,
				minSize=(20, 20),
				flags = cv2.cv.CV_HAAR_SCALE_IMAGE
			)
		print len(torso_list)
		not_valid_torsos = []
		for i in reversed(xrange(len(torso_list))): 
			x_torso, y_torso, w_torso, h_torso = torso_list[i]
			# center_torso = (x_torso + (w_torso / 2), y_torso + (h_torso / 2))
			valid_torso = False 
			for bb in bbs: 
				x, y, w_bb, h_bb = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
				if x_torso >= x and x_torso + w_torso < x + w_bb \
					and y_torso >= y and y_torso + h_torso <= y + h_bb: 
					valid_torso = True

			not_valid_torsos.append(i)
			# if not valid_torso: 
			# 	del torso_list[i]	

		# torso_list = [torso for i,torso in enumerate(torso_list) if i not in not_valid_torsos]		

		print "Found {0} torsos!".format(len(torso_list))

		# Draw a rectangle around the faces
		for (x, y, w, h) in torso_list:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
		
		cv2.imshow("torsos found", image)
		cv2.waitKey(0)	

		return torso_list, image

