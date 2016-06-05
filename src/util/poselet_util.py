import cv2
import os
import scipy.io as sio
import matlab.engine
import matlab
import numpy as np
import pandas as pd

from FaceExtractor import FaceExtractor

def init(eng):
	eng.addpath(r'/Users/tariq/Dev/School/VisualSentiment/poselets_matlab_april2013/detector/')
	eng.addpath(r'/Users/tariq/Dev/School/VisualSentiment/poselets_matlab_april2013/detector/categories')
	eng.addpath(r'/Users/tariq/Dev/School/VisualSentiment/poselets_matlab_april2013/detector/poselet_detection')

def getTorsos(img_path="../../data/groupdataset_release/images/all/5164048347_af12243081_z.jpg"):
	eng = matlab.engine.start_matlab()
	init(eng)

	# config = eng.init()
	# print config

	# get model
	# model = matlab.object(sio.loadmat('../../poselets_matlab_april2013/data/person/model.mat'))
	# model = eng.load('../../poselets_matlab_april2013/data/person/model.mat')

	# get image
	img = cv2.imread(img_path)
	img_mat = matlab.uint8(img.tolist())
	# print img

	# # testarr = np.array([1,2,3])
	# # testlist = testarr.tolist()
	# # print testlist
	# if not os.path.isfile('../cache/torso_bounds.npy'):
	torso_bounds, torso_scores = eng.detect_objects_in_image_python(img_mat, nargout=2)
	# 	np.save('../cache/torso_bounds.npy', torso_bounds)
	# 	np.save('../cache/torso_scores.npy', torso_scores)
	# else:
	# 	torso_bounds = np.load('../cache/torso_bounds.npy')
	# 	torso_scores = np.load('../cache/torso_scores.npy')

	torso_bounds = np.array(torso_bounds)
	torso_scores = np.array(torso_scores)
	torso_bounds = torso_bounds.T

	# show torsos on image
	for i in xrange(torso_bounds.shape[0]):
		print torso_scores[i]
		x, y, w, h = torso_bounds[i,:]
		cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

	cv2.imshow("Torsos in image", img)
	cv2.waitKey(0)




	eng.quit()

FACE_POSELET_IDS = [7,16,19,22,24,25,28,30,34,35,45,48,51,53,63,72,74,80,83,100,105,112,115,119,129]
POSELET_SCORE_THRESH = 0.5
def getHeadPoselets(img_path="../../data/groupdataset_release/images/all/5164048347_af12243081_z.jpg",\
 poselets_folder="../../data/groupdataset_release/all_poselets/"):
	"""
    Runs face detector on relevant poselets. Returns list of faces for given image. 
    """

    # read in the image and get the actual image name
	img = cv2.imread(img_path)

	img_name = os.path.split(img_path)[-1]
	img_name = img_name[:-4]

	# retrieve relevant poselets
	poselet_tag = '_poselets.csv'
	poselet_path = os.path.join(poselets_folder, img_name + poselet_tag)
	poselets = pd.read_csv(poselet_path).as_matrix()
	# print type(poselets)

	# find all poselets corresponding to run face extractor on
	face_poselets = []

	for i in range(len(poselets)):
		poselet = poselets[i]
		poselet_score = poselet[-1]

		if poselet_score > POSELET_SCORE_THRESH:
			# check if it's a poselet we want
			poselet_id = int(poselet[-2])

			if poselet_id in FACE_POSELET_IDS:
				face_poselets.append(poselet[:4])

	return face_poselets

	# # run face extractor on all faces
	# face_extractor = FaceExtractor()
	# faces_from_poselets = []

	# for i in range (len(face_poselets)):
	# 	# get the poselet 
	# 	x,y,w,h = face_poselets[i]
	# 	x = int(x)
	# 	y = int(y)
	# 	w = int(w)
	# 	h = int(h)

	# 	cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 1)

	# 	poselet = img[y:y+h, x:x+w]

	# 	faces_list = face_extractor.detect_faces(poselet)
	# 	faces_from_poselets.extend(faces_list)

	
	# for i in range(len(faces_from_poselets)):
	# 	x, y, w, h = faces_from_poselets[i]
	# 	cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
		
	# cv2.imshow("Faces found", img)
	# cv2.waitKey(0)









if __name__ == '__main__':
	getHeadPoselets()

