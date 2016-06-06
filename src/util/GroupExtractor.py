import cv2
import numpy as np


class GroupExtractor: 

	def __init__(self): 
		pass


	def threeD_estimation(face_list): 
		heights = []
		for i,face in face_list: 
			x, y, w, h = face 
			heights.append((h,i))

		sorted_heights = sorted(heights, key=itemgetter(0,1), reverse=True) #smallest to largest 
		sorted_heights 

		z_coordinates = []







