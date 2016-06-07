import numpy as np 
import os
import cv2

from xml.dom import minidom

def test_orientation_svm():
	# get all testing images

	# get silhouettes of testing images

	# run SIFT on silhouettes

	# run HOG on regular images

	# combine feature vecs

	# get corresponding labels

	# test and report accuracy
	pass

def train_orientation_svm():
	# get all training images

	# get silhouettes of training images

	# run SIFT on silhouettes

	# run HOG on regular images

	# combine feature vecs

	# create input feature matrix with corresponding labels

	# train
	pass

def create_filename_orient_dict(tud_path='../../../data/TUD', tud_type = 'train'):
	"""
    Extracts orientations for a given data set type (train, test, validate) and creates
    a dict from img_name -> orientation 
    """


	# build the train dict
	# train_dict = {}
	img_folder = os.path.join(tud_path, tud_type)
	img_filenames = [f for f in os.listdir(img_folder) if 'DS_Store' not in f]
	print len(img_filenames)

	# get corresponding training annotations
	annotation_type = '_' + tud_type
	train_annotation_files = [f for f in os.listdir(tud_path) if annotation_type in f]
	# print train_annotation_files

	all_orientations = []
	for file in train_annotation_files:
		fullpath = os.path.join(tud_path,file)
		# annotation_xml = xml.etree.ElementTree.parse(fullpath).getroot()
		annotation_xml = minidom.parse(fullpath)
		sil_list = annotation_xml.getElementsByTagName('silhouette')
		for sil in sil_list:
			orientation = sil.getElementsByTagName('id')
			orientation = orientation[0]
			orientation = orientation.firstChild.nodeValue
			all_orientations.append(orientation)

	return dict(zip(img_filenames, all_orientations))


if __name__ == '__main__':
    create_filename_orient_dict()