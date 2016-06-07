import cv2
import numpy as np

FACE_HEIGHT = 0.17
CAM_FOC_LENGTH = 500
SCALE_PARAM = FACE_HEIGHT * CAM_FOC_LENGTH
DOT_PROD_SCALING = 10

class GroupExtractor: 

	def __init__(self): 
		pass

	def get_face_depths(face_matrix): 
		depths = np.zeros(face_list.shape[0])
		for i,face in face_matrix: 
			_, _, _, h = face 
			depth = SCALE_PARAM / h
			depths[i] = depth
		return depths

	def get_3D_coordinates(face_matrix):
		depths = get_face_depths(face_matrix)
		n_faces = face_matrix.shape[0]
		threeD = np.zeros((n_faces, 3))
		for i in xrange(n_faces):
			x, y, w, h = face_matrix[i,:]
			center = (x + w / 2, y + h / 2)
			threeD[i,:] = np.array([center[0], center[1], depths[i]])
		return threeD

	def get_clusters(threeD, orientations):
		n_people = threeD.shape[0]
		max_k = int(np.ceil(n_people / 2.0))

		for k in xrange(1, max_k+1):
			cent_idxs = np.random.choice(np.arange(n_people), size=k)
			cents = threeD[cents_idxs,:]
