import numpy as np 
import os
import cv2
import sys	

# from xml.dom import minidom
import xml.etree.ElementTree as ET
sys.path.insert(0, '../.')
from SilhouetteExtractor import SilhouetteExtractor
from image_util import *

TUD_PATH = '../../../data/TUD'

NORMALIZED_SIZE = 300

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
	# silExtractor.get_silhouettes()
	run_extraction_again = False
	run_1 = False

	# get all training images
	tud_type = 'train'
	train_dict = create_filename_orient_dict(TUD_PATH, tud_type)

	img_list = []
	img_names = []

	FEATURE_VEC_SIZE = 2494800

	X = []
	Y = []
	if not os.path.isfile('../../cache/orientation_features2.npy') or run_extraction_again: 
		X = np.zeros((len(train_dict), FEATURE_VEC_SIZE))
		for i,img_name in enumerate(train_dict.keys(), 2500):
			if run_1 and i == 2500:
				break
		# for img_label_pts in train_dict.keys():
			img_names.append(img_name)
			img_path = os.path.join(TUD_PATH,tud_type,img_name)
			img = cv2.imread(img_path)

			gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# print NORMALIZED_SIZE
			img = cv2.resize(gray_img, (NORMALIZED_SIZE, NORMALIZED_SIZE))

			# resize to 100 by 100
	 		hog = cv2.HOGDescriptor()
	 		# im = cv2.imread(sample)
			h = hog.compute(img)
			print i
			# print X.shape
			X[i,:] = h.T

		Y = train_dict.values()
		# print Y
		np.save('../../cache/orientation_features1', X)
		np.save('../../cache/orientation_labels',Y)
	else:
		np.load('../../cache/orientation_features', X)
		np.load('../../cache/orientation_labels',Y)

	# for img_name,orientation in train_dict.iteritems():
	# for img_label_pts in train_dict.keys():
		# img_names.append(img_name)
		# img_path = os.path.join(TUD_PATH,tud_type,img_name)
		# img = cv2.imread(img_path)

		# gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# # print NORMALIZED_SIZE
		# img = cv2.resize(gray_img, (NORMALIZED_SIZE, NORMALIZED_SIZE))

		# # resize to 100 by 100
 	# 	hog = cv2.HOGDescriptor()
 	# 	# im = cv2.imread(sample)
		# h = hog.compute(img)
		# print feature_matrix.shape
		# feature_matrix[i,:] = h.T

		# print h.T.shape



		# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# sift = cv2.SIFT()
		# kp = sift.detect(gray,None)
		# kp, des = sift.compute(gray,kp)
		# bow.add(des)
		# print des.shape
		# print h.shape

		# feature_matrix_dict[]

		# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		# sift = cv2.features2d.SIFT_create()

		# img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imshow('Faces', img)
		# cv2.waitKey(0)
		# img_list.append(img)

	# dictionary = bow.cluster()
	# print dictionary




		# label, _ = label_pts

		# person_bbs, torso_bbs = get_bbs(img_name) 
		# for pt in pts:
		# 	cv2.circle(img, pt, 10, (0, 255, 0), 2) 	
		# cv2.rectangle(img, , (0, 255, 0), 2)
		# cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 
		# print img
		# cv2.imshow('silhouettes', img)
		# cv2.waitKey(0)

	# get silhouettes of training images
	# people_bbs, torso_bbs = 
	# print get_all_silhouettes(tud_type)
	# print len(img_names)
	# print len(img_list)
	# sil_images, img_names = get_all_silhouettes(tud_type, img_list, img_names)
	# print len(sil_images)
	# print len(img_names)

	# face_bbs = np.zeros(people_bbs.shape)

	# run SIFT on silhouettes
	# for sil_image in sil_images:
		# img = cv2.imread('home.jpg')
		
   		# cv2.imwrite(,img)



	# run HOG on regular images

	# combine feature vecs

	# create input feature matrix with corresponding labels

	# train

def get_all_bbs(tud_type):
	people_bbs_path = os.path.join(TUD_PATH, tud_type, 'all_people')
	torsos_bbs_path = os.path.join(TUD_PATH, tud_type, 'all_torsos')
	# poselets_bbs_path = os.path.join(TUD_PATH, tud_type, 'all_poselets')
	people_filenames = [filename for filename in os.listdir(people_bbs_path) if '.DS_Store' not in filename]
	torso_filenames = [filename for filename in os.listdir(torsos_bbs_path) if '.DS_Store' not in filename]

	# print len(people_filenames)

	people_bbs = np.zeros((len(people_filenames),5))
	for i, people_filename in enumerate(people_filenames):
		# read in as array	
		people_bbs_path_for_image = os.path.join(people_bbs_path, people_filename)
		all_bbs_for_image = pd.read_csv(people_bbs_path_for_image).as_matrix()

		all_bbs_for_image = all_bbs_for_image[all_bbs_for_image[:,-1].argsort()[::-1]]
		# all_bbs_for_image.sort(order='s')
		# print all_bbs_for_image
		# get the top 5 scoring bb
		people_bbs[i,:] = all_bbs_for_image[0]
		# return as a list

	torso_bbs = np.zeros((len(torso_filenames),5))
	for i, torso_filename in enumerate(torso_filenames):
		# read in as array	
		torso_bbs_path_for_image = os.path.join(torsos_bbs_path, torso_filename)
		all_bbs_for_image = pd.read_csv(torso_bbs_path_for_image).as_matrix()

		all_bbs_for_image = all_bbs_for_image[all_bbs_for_image[:,-1].argsort()[::-1]]
		# all_bbs_for_image.sort(order='s')
		# print all_bbs_for_image
		# get the top scoring bb
		torso_bbs[i,:] = all_bbs_for_image[0]
		# return as a list

	return people_bbs, torso_bbs





def get_all_silhouettes(tud_type, img_list='', img_names=''):
	"""
	Get highest score torso from the image. Get highe 

	"""

	silExtractor = SilhouetteExtractor()
	# get all boundinb boxes for each image
	people_bbs, torso_bbs = get_all_bbs(tud_type)

	# print people_bbs.shape
	# print len(img_list)
	print torso_bbs.shape

	silhoutted_image_names = []
	sil_images = []

	for i, people_bb in enumerate(people_bbs):
		face_bbs = np.zeros((0,0))
		torso_bb = torso_bbs[i].astype('uint8')
		torso_bb = np.array([torso_bb[:-1]])
		people_bb = np.array([people_bb[:-1]])

		img = img_list[i]
		x, y, w, h = people_bb[0].astype('uint8')
		cv2.rectangle(img, (x, y), (int(x+w), int(y+h)), (0, 255, 0), 2)
		cv2.imshow("Faces found", img)
		cv2.waitKey(0)

		matched_list = bb_matching(img_list[i], people_bb, face_bbs, torso_bb)
		# print len(matched_list)

		if len(matched_list) != 0:
			# get corresponding image
			img = img_list[i]

			# run grab cut on the corresponding image. Get back silhoutted image
			sils_for_image = silExtractor.get_silhouetted_images(img, matched_list)
			sil_images.append(sils_for_image[0])

			# add to list of feature images
			silhoutted_image_names.append(img_names[i])

	return sil_images, silhoutted_image_names



def create_filename_orient_dict(tud_path='../../../data/TUD', tud_type = 'train'):
	"""
    Extracts orientations for a given data set type (train, test, validate) and creates
    a dict from img_name -> orientation 
    """

    # root = ET.parse(".xml")
    # logentries = root.findall("logentry")
    # content = ""

    # for logentry in logentries:
    #     date = logentry.find("date").text
    #     content += date + '\n '
    #     msg = logentry.find("msg")
    #     if msg is not None:
    #         content += "   Comments: \n        " + msg.text + '\n\n'
    #     else:
    #         content += "No comment made."

    # print content

	# build the train dict
	# train_dict = {}
	img_folder = os.path.join(tud_path, tud_type)
	# img_filenames = []
	# for f in os.lisdir(img_folder):
	# 	if 'DS_Store' not in f and  '.png' in f:
	# 		img_filenames.append(f)
	img_filenames = [f for f in os.listdir(img_folder) if 'DS_Store' not in f and '.png' in f]
	# print len(img_filenames)

	# get corresponding training annotations
	annotation_type = '_' + tud_type
	train_annotation_files = [f for f in os.listdir(tud_path) if annotation_type in f]
	# print train_annotation_files

	# all_orientations = []
	filename_orientation_dict = {}

	# num_annotations = 

	for file in train_annotation_files:
		fullpath = os.path.join(tud_path,file)
		# annotation_xml = xml.etree.ElementTree.parse(fullpath).getroot()
		# annotation_xml = minidom.parse(fullpath)

		annotation_xml = ET.parse(fullpath)
		# print annotation_xml
		annotation_list = annotation_xml.findall('annotation')
		for annotation in annotation_list:
			img_name = annotation.find('image').find('name').text
			img_name = img_name.split('/')
			img_name = img_name[1]

			annorect = annotation.find('annorect')

			sil = annorect.find('silhouette')
			if sil == None:
				sil = 0
			else:
				sil = int(sil.find('id').text)

			# sil = sil[0].findall('id')
			# sil = sil[0].text
			# print sil 
			# pts = annorect.find('annopoints').findall('point')
			# img_pts = []
			# for pt in pts:
			# 	x = pt.find('x').text
			# 	y = pt.find('y').text
			# 	img_pts.append((int(x),int(y)))

			# label_pts = sil
			filename_orientation_dict[img_name] = sil 
			# filename_orientation_dict[img_name] += sil

	# print len(img_filenames)
	# print len(filename_orientation_dict)
	# print len(img_filenames)


  
			

		# for sil in sil_list:
		# 	orientation = sil.find('id')
		# 	print orientation.text
			# orientation = orientation[0]
			# orientation = orientation.firstChild.nodeValue
			# all_orientations.append(orientation)


		# for sil in sil_list:
		# 	orientation = sil.getElementsByTagName('id')
		# 	orientation = orientation[0]
		# 	orientation = orientation.firstChild.nodeValue
		# 	all_orientations.append(orientation)

		# for testing. Get annotation points and draw them on the image
		# annotation_pts = annotation_xml.getElementsByTagName('annopoints')
		# for pts_for_person in annotation_pts:
		# 	pt = pts_for_person.getElementsByTagName('point')
		# 	# print pt
		# 	print pt.firstChild.childNodes
			# print pt[2].firstChild.nodeValue
			# x = pt.getElementsByTagName('x')
			# y = pt.getElementsByTagName('y')
			# break

		# break

	return filename_orientation_dict


if __name__ == '__main__':
    train_orientation_svm()