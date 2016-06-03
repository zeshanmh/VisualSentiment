import cv2
import os
import scipy.io as sio
import matlab.engine
import matlab
import numpy as np

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
	if not os.path.isfile('../cache/torso_bounds.npy'):
		torso_bounds, torso_scores = eng.detect_objects_in_image_python(img_mat, nargout=2)
		np.save('../cache/torso_bounds.npy', torso_bounds)
		np.save('../cache/torso_scores.npy', torso_scores)
	else:
		torso_bounds = np.load('../cache/torso_bounds.npy')
		torso_scores = np.load('../cache/torso_scores.npy')

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

if __name__ == '__main__':
	getTorsos()

