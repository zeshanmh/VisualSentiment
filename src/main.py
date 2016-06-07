import sys
import os
import cv2
import sklearn
import analysis
import numpy as np
sys.path.insert(0, './util')
import image_util

# from FaceExtractor import FaceExtractor
# from EmotionExtractor import EmotionExtractor
# from TorsoExtractor import TorsoExtractor
# from sklearn import svm
# from sklearn.decomposition import PCA
# from sklearn.externals import joblib
from EmotionSVM import EmotionSVM
from feature_extractors import *

def main():
	##Torso Extraction##
	# img_path = "../data/groupdataset_release/images/4940922642_5dab04b030_o.jpg"
	# torso_extractor = TorsoExtractor()
	# torso_list, image = torso_extractor.detect_torsos(img_path)
	

	# ## Face Extraction ## 
	# # img_path = "../data/groupdataset_release/images/Library3.jpg"
	# # face_extractor = FaceExtractor()
	# # faces_lists, image = face_extractor.detect_faces(img_path)
	# # for face_list in faces_lists: 
	# # 	for (x,y,w,h) in face_list: 

	# extract_faces = False
	# extract_missed_faces = False 
	# if extract_faces: 
	# 	src_path = '../data/GENKI-R2009a/Subsets/GENKI-4K/files'
	# 	dest_path = './cache/GENKI_faces'
	# 	image_util.extract_GENKI_faces(src_path, dest_path)

	# if extract_missed_faces: 
	# 	src_path = '../data/GENKI-R2009a/Subsets/GENKI-4K/files'
	# 	dest_path = './cache/GENKI_faces/GENKI_faces_looser_bounds'
	# 	image_util.extract_missed_faces(dest_path)

	img_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Images_Reduced.txt"
	labels_path = "../data/GENKI-R2009a/Subsets/GENKI-4K/GENKI-4K_Labels_Reduced.txt"
	img_path2 = '../data/groupdataset_release/images'
	faces_path = '../data/groupdataset_release/faces'

	train_again = False
	if train_again:
		svm = EmotionSVM(img_path, labels_path, img_path2, 'sad', dump=True)
		svm.train()
		# svm = train_smile_extractor(img_path, labels_path)	
		# joblib.dump(svm, 'svm_model.pkl')
	else: 
		pass
		# print 'Loading svm...'
		# svm = EmotionSVM(img_path, labels_path, img_path2, 'smile', fit=False)
		# all_face_features = get_all_face_features(img_path2, faces_path, svm)
		# print all_face_features.shape
		# np.save('../data/groupdataset_release/face_features.npy', all_face_features)

	poselet_path = '../data/groupdataset_release/all_poseletes_hq'
	all_poselet_features = get_all_poselet_features(poselet_path)
	print all_poselet_features.shape
	np.save('../data/groupdataset_release/poselet_features.npy', all_poselet_features)


	# X = get_emotion_vector(svm)
	# test_path = ''
	# filenames = os.listdir(test_path)
	

		
	# img_path2 = '../data/groupdataset_release/images/all/466491971_b3bfbce419_o.jpg'
	# img_path2 = '../data/groupdataset_release/images/all/419925_10150597648651087_1342010291_n.jpg'
	# filenames = os.listdir(img_path2)
	# filenames = [os.path.join(img_path2, filename) for filename in filenames]

	# for path in filenames:
	# 	#or '01-breeze-outdoor-dining.jpg' in img_path2:  
	# 	if '.DS' in path: 
	# 		continue

	# 	# faces_list, smile_features, scores = predict_smiles(path, svm)
	# 	faces_list, smile_features, predictions, scores = svm.predict(path)


	# 	all_faces = [face for face_list in faces_list for face in face_list]
	# 	image = cv2.imread(path)
	# 	for i,face in enumerate(all_faces): 
	# 		x,y,w,h = face  
	# 		# prediction = np.argmax(scores[i,:])
	# 		print "coordinates:", face
	# 		print "score %d:" % (scores[i])
	# 		print "prediction for face %d: %d" % (i,predictions[i])

	# 		if predictions[i] == 0:
	# 			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	# 		else: 
	# 			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

	# 	cv2.imshow("blah", image)
	# 	cv2.waitKey(0)

	# 	print faces_list
	# 	print predictions


if __name__ == '__main__':
	main()
